"""Stress tests trying to BREAK the system: chaos testing and thundering herd coalescing."""

import os
import random
import threading
import time
import pytest

from miniedge.clock import Clock
from miniedge.edge import EdgeNode
from miniedge.errors import (
    LeaseExpired,
    LockTimeout,
    RateLimited,
    StaleFencingToken,
    TransientBackendError,
    VersionConflict,
)
from miniedge.locks import LockService
from miniedge.object_store import ObjectStore
from miniedge.ratelimit import PerClientLimiter, TokenBucket


@pytest.mark.slow
def test_chaos_multi_node_concurrent_workload() -> None:
    """Chaos test: 2 edge nodes, 40 client threads doing random reads/writes over 10 files for 5s.

    Invariants verified:
    1. No exceptions other than expected error types.
    2. No deadlock: guarded by a watchdog timeout.
    3. Final data of each file matches the ground-truth model maintained under the lock.
    """
    seed = int(os.environ.get("MINIEDGE_TEST_SEED", random.randint(1, 1_000_000_000)))
    print(f"\n[test_chaos] Execution seed: {seed}")
    master_rng = random.Random(seed)

    store = ObjectStore(
        failure_rate=0.1,
        base_latency=0.001,
        latency_jitter=0.003,
        seed=master_rng.randint(1, 1_000_000),
    )
    locks = LockService(default_ttl=2.0)
    backend_limiter_1 = TokenBucket(capacity=10000, refill_rate=10000)
    backend_limiter_2 = TokenBucket(capacity=10000, refill_rate=10000)
    client_limiter_1 = PerClientLimiter(capacity=10000, refill_rate=10000)
    client_limiter_2 = PerClientLimiter(capacity=10000, refill_rate=10000)

    node1 = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=backend_limiter_1,
        client_limiter=client_limiter_1,
        lease_ttl=2.0,
    )
    node2 = EdgeNode(
        node_id="node_2",
        store=store,
        locks=locks,
        backend_limiter=backend_limiter_2,
        client_limiter=client_limiter_2,
        lease_ttl=2.0,
    )

    num_files = 10
    file_size = 256  # Pre-allocate files so file length does not change mid-chaos test
    files = [f"chaos_{i}.bin" for i in range(num_files)]
    for f in files:
        node1.write("init", f, 0, b"\x00" * file_size)

    # In-memory ground-truth model and per-file write serialization locks
    model = {f: bytearray(b"\x00" * file_size) for f in files}
    file_locks = [threading.Lock() for _ in range(num_files)]

    expected_error_types = (
        TransientBackendError,
        VersionConflict,
        LeaseExpired,
        RateLimited,
        LockTimeout,
        StaleFencingToken,
    )
    unexpected_errors: list[tuple[str, Exception]] = []
    unexpected_lock = threading.Lock()

    stop_event = threading.Event()

    def client_worker(node: EdgeNode, client_id: str, worker_seed: int) -> None:
        rng = random.Random(worker_seed)
        while not stop_event.is_set():
            file_idx = rng.randint(0, num_files - 1)
            file_path = files[file_idx]
            operation = rng.choice(["read", "read", "write"])

            if operation == "read":
                offset = rng.randint(0, file_size - 10)
                length = rng.randint(1, file_size - offset)
                try:
                    node.read(client_id, file_path, offset, length)
                except expected_error_types:
                    pass
                except Exception as exc:
                    with unexpected_lock:
                        unexpected_errors.append((client_id, exc))
            else:
                length = rng.randint(5, 30)
                offset = rng.randint(0, file_size - length)
                payload = bytes(rng.randint(65, 90) for _ in range(length))

                with file_locks[file_idx]:
                    try:
                        node.write(client_id, file_path, offset, payload)
                        # When write commits, update the reference model
                        model[file_path][offset : offset + length] = payload
                    except expected_error_types:
                        pass
                    except Exception as exc:
                        with unexpected_lock:
                            unexpected_errors.append((client_id, exc))

    threads: list[threading.Thread] = []
    for i in range(20):
        t1 = threading.Thread(
            target=client_worker,
            args=(node1, f"c1_{i}", master_rng.randint(1, 1_000_000)),
        )
        t2 = threading.Thread(
            target=client_worker,
            args=(node2, f"c2_{i}", master_rng.randint(1, 1_000_000)),
        )
        threads.extend([t1, t2])

    for t in threads:
        t.start()

    # Run chaos workload for 5 seconds
    time.sleep(5.0)
    stop_event.set()

    # Watchdog timeout: join all threads within 10 seconds to detect deadlocks
    stuck_threads: list[str] = []
    for t in threads:
        t.join(timeout=10.0)
        if t.is_alive():
            stuck_threads.append(t.name)

    assert not stuck_threads, f"Deadlock detected! Threads stuck under seed {seed}: {stuck_threads}"
    assert not unexpected_errors, f"Unexpected errors under seed {seed}: {unexpected_errors}"

    # Verify final data of each file matches the ground-truth model
    mismatches: list[str] = []
    for f in files:
        expected = bytes(model[f])
        data1 = node1.read("verifier", f, 0, file_size)
        data2 = node2.read("verifier", f, 0, file_size)
        if data1 != expected or data2 != expected:
            mismatches.append(f"{f}: expected={expected!r}, node1={data1!r}, node2={data2!r}")

    assert not mismatches, f"Model mismatches under seed {seed}:\n" + "\n".join(mismatches)


def test_thundering_herd_coalescing() -> None:
    """Thundering herd test: 200 threads reading one cold key causes exactly 1 backend fetch."""
    seed = int(os.environ.get("MINIEDGE_TEST_SEED", random.randint(1, 1_000_000_000)))
    rng = random.Random(seed)

    # 10ms simulated latency to give all 200 threads time to pile up on the same in-flight fetch
    store = ObjectStore(base_latency=0.01, seed=rng.randint(1, 1_000_000))
    locks = LockService(default_ttl=10.0)
    b_lim = TokenBucket(capacity=1000, refill_rate=1000)
    c_lim = PerClientLimiter(capacity=1000, refill_rate=1000)
    node = EdgeNode("node_herd", store=store, locks=locks, backend_limiter=b_lim, client_limiter=c_lim)

    file_data = bytes(rng.randint(0, 255) for _ in range(1024))
    store.put("herd.bin#meta", str(len(file_data)).encode("utf-8"))
    store.put("herd.bin#0", file_data)

    # Pre-cache #meta so all 200 readers coalesce strictly on chunk 0
    node._cache.put("herd.bin#meta", str(len(file_data)).encode("utf-8"), version=1, dirty=False)

    store.reset_stats()
    num_threads = 200
    results: list[bytes] = [b""] * num_threads
    errors: list[Exception] = []

    def reader(idx: int) -> None:
        try:
            results[idx] = node.read(f"reader_{idx}", "herd.bin", 0, 1024)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Errors during thundering herd under seed {seed}: {errors}"
    for idx, r in enumerate(results):
        assert r == file_data, f"Reader {idx} received incorrect data under seed {seed}"

    store_stats = store.stats()
    assert store_stats["gets"] == 1, (
        f"Expected exactly 1 backend get, got {store_stats['gets']} under seed {seed}"
    )
