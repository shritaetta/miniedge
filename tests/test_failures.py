"""Tests attempting to break the system: deliberate failures, lease expiry mid-write, and throttling."""

import os
import random
import threading
import time
import pytest

from miniedge.edge import EdgeNode
from miniedge.errors import (
    LeaseExpired,
    StaleFencingToken,
    Throttled,
    TransientBackendError,
)
from miniedge.locks import LockService
from miniedge.object_store import ObjectRecord, ObjectStore
from miniedge.ratelimit import PerClientLimiter, TokenBucket


def test_lease_expiry_mid_write_using_slow_store() -> None:
    """A deliberately slow store delays put beyond lease TTL; store fencing rejects the stale write."""
    seed = int(os.environ.get("MINIEDGE_TEST_SEED", random.randint(1, 1_000_000_000)))
    rng = random.Random(seed)

    store = ObjectStore(seed=rng.randint(1, 1_000_000))
    # Short TTL so expiry happens quickly in real-time
    ttl = 0.15
    locks = LockService(default_ttl=ttl)
    b1 = TokenBucket(1000, 1000)
    b2 = TokenBucket(1000, 1000)
    c1 = PerClientLimiter(1000, 1000)
    c2 = PerClientLimiter(1000, 1000)

    node1 = EdgeNode("node_1", store=store, locks=locks, lease_ttl=ttl, backend_limiter=b1, client_limiter=c1)
    node2 = EdgeNode("node_2", store=store, locks=locks, lease_ttl=ttl, backend_limiter=b2, client_limiter=c2)

    # Initial file setup
    file_path = "slow_expiry.bin"
    node1.write("init", file_path, 0, b"initial_state")

    real_put = store.put
    node1_in_put = threading.Event()
    node2_committed = threading.Event()

    def slow_put(key: str, data: bytes, expected_version=None, fencing_token=None) -> int:
        # Intercept node 1's subsequent write attempt (fencing token == 2)
        if fencing_token == 2:
            node1_in_put.set()
            # Wait until node 2 has acquired the lock, committed, and advanced the fencing token
            node2_committed.wait(timeout=3.0)
        return real_put(key, data, expected_version=expected_version, fencing_token=fencing_token)

    store.put = slow_put  # type: ignore[assignment]

    node1_errors: list[Exception] = []

    def writer1() -> None:
        try:
            node1.write("c1", file_path, 0, b"stale_data_node1")
        except Exception as exc:
            node1_errors.append(exc)

    t1 = threading.Thread(target=writer1)
    t1.start()

    # Wait for node 1 to enter slow put
    assert node1_in_put.wait(timeout=2.0), f"Node 1 failed to enter put under seed {seed}"

    # Sleep longer than node 1's lease TTL (0.15s) so the lease expires
    time.sleep(0.20)

    # Node 2 acquires the exclusive lease (awarded token 3) and commits its write
    node2.write("c2", file_path, 0, b"committed_data_node2")
    node2_committed.set()

    t1.join(timeout=3.0)

    # Node 1's write must be rejected by either LeaseExpired (edge check) or StaleFencingToken (store check)
    assert len(node1_errors) == 1, f"Expected 1 error from node 1, got {node1_errors} under seed {seed}"
    assert isinstance(node1_errors[0], (LeaseExpired, StaleFencingToken)), (
        f"Unexpected exception type under seed {seed}: {type(node1_errors[0])}"
    )

    # Verify backend and both nodes observe node 2's committed data, not node 1's stale data
    assert node2.read("c2", file_path, 0, 30) == b"committed_data_node2"
    assert node1.read("c1", file_path, 0, 30) == b"committed_data_node2"


def test_throttling_limiter_on_vs_off() -> None:
    """Backend throttles at N req/sec: with limiter ON, error rate is ~0; with limiter OFF, errors occur."""
    seed = int(os.environ.get("MINIEDGE_TEST_SEED", random.randint(1, 1_000_000_000)))
    rng = random.Random(seed)

    n_limit = 20

    # 1. Limiter OFF: unconstrained egress burst floods the store and triggers Throttled exceptions
    store_off = ObjectStore(max_calls_per_second=n_limit, seed=rng.randint(1, 1_000_000))
    limiter_off = TokenBucket(capacity=1000, refill_rate=1000)
    node_off = EdgeNode(
        "node_off",
        store=store_off,
        locks=LockService(),
        backend_limiter=limiter_off,
        client_limiter=PerClientLimiter(1000, 1000),
    )

    # Populate objects directly into store_off so sliding window timestamps are completely clean
    for i in range(40):
        store_off._objects[f"item_{i}#meta"] = ObjectRecord(b"10", 1, 0)
        store_off._objects[f"item_{i}#0"] = ObjectRecord(b"1234567890", 1, 0)

    errors_off: list[Exception] = []

    def burst_worker_off(idx: int) -> None:
        try:
            node_off.read(f"client_{idx}", f"item_{idx}", 0, 10)
        except Exception as exc:
            errors_off.append(exc)

    threads_off = [threading.Thread(target=burst_worker_off, args=(i,)) for i in range(35)]
    for t in threads_off:
        t.start()
    for t in threads_off:
        t.join(timeout=5.0)

    # Assert errors occurred when limiter was off
    throttled_count_off = sum(1 for e in errors_off if isinstance(e, Throttled))
    assert throttled_count_off > 0, (
        f"Expected Throttled errors with limiter off, got none under seed {seed}"
    )

    # 2. Limiter ON: capacity C and refill rate R chosen such that C + R * 1.0 <= N
    # This guarantees that in any 1-second sliding window, calls to ObjectStore never exceed N
    store_on = ObjectStore(max_calls_per_second=n_limit, seed=rng.randint(1, 1_000_000))
    limiter_on = TokenBucket(capacity=5, refill_rate=14)
    node_on = EdgeNode(
        "node_on",
        store=store_on,
        locks=LockService(),
        backend_limiter=limiter_on,
        client_limiter=PerClientLimiter(1000, 1000),
    )

    for i in range(40):
        store_on._objects[f"item_{i}#meta"] = ObjectRecord(b"10", 1, 0)
        store_on._objects[f"item_{i}#0"] = ObjectRecord(b"1234567890", 1, 0)

    errors_on: list[Exception] = []

    def paced_worker_on(idx: int) -> None:
        try:
            node_on.read(f"client_{idx}", f"item_{idx}", 0, 10)
        except Exception as exc:
            errors_on.append(exc)

    threads_on = [threading.Thread(target=paced_worker_on, args=(i,)) for i in range(30)]
    for t in threads_on:
        t.start()
    for t in threads_on:
        t.join(timeout=5.0)

    # Assert error rate is ~0 (zero unhandled Throttled errors) with limiter on
    throttled_count_on = sum(1 for e in errors_on if isinstance(e, Throttled))
    assert throttled_count_on == 0, (
        f"Expected 0 Throttled errors with limiter on, got {throttled_count_on} under seed {seed}"
    )
    assert not errors_on, f"Unexpected errors with limiter on under seed {seed}: {errors_on}"


def test_backend_outage_then_recovery() -> None:
    """Backend outage causes uncached requests to fail, but system recovers smoothly once outage clears."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    b_lim = TokenBucket(1000, 1000)
    c_lim = PerClientLimiter(1000, 1000)
    node = EdgeNode("node_outage", store=store, locks=locks, backend_limiter=b_lim, client_limiter=c_lim)

    # Populate and cache file
    node.write("c1", "resilient.txt", 0, b"cached_payload")
    assert node.read("c1", "resilient.txt", 0, 14) == b"cached_payload"

    # Simulate total backend outage
    store.set_outage(True)
    try:
        # Cached reads succeed without contacting the store
        assert node.read("c1", "resilient.txt", 0, 14) == b"cached_payload"

        # Uncached read attempts backend fetch and raises TransientBackendError
        with pytest.raises(TransientBackendError, match="outage in progress"):
            node.read("c1", "uncached_missing.txt", 0, 10)
    finally:
        # Outage cleared
        store.set_outage(False)

    # Node recovers immediately and serves new reads/writes
    node.write("c1", "post_outage.txt", 0, b"recovered")
    assert node.read("c1", "post_outage.txt", 0, 9) == b"recovered"
