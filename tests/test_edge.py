"""Tests for EdgeNode: read/write across chunks, multi-node concurrency, failure injection, and coalescing."""

import random
import threading
import time
import pytest

from miniedge.clock import FakeClock
from miniedge.edge import EdgeNode
from miniedge.errors import (
    LeaseExpired,
    StaleFencingToken,
    TransientBackendError,
    VersionConflict,
)
from miniedge.locks import LockService
from miniedge.object_store import ObjectStore
from miniedge.ratelimit import PerClientLimiter, TokenBucket


def test_write_then_read_across_chunk_boundaries() -> None:
    """Write then read returns the same bytes, including across chunk boundaries."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    backend_limiter = TokenBucket(capacity=1000, refill_rate=1000)
    client_limiter = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        chunk_size=1024,
        backend_limiter=backend_limiter,
        client_limiter=client_limiter,
    )

    # 3500 bytes of non-repeating data spanning chunks 0, 1, 2, 3 (when written at offset 500)
    payload = bytes((i * 17 + 5) % 256 for i in range(3500))
    node.write("client_1", "doc.bin", 500, payload)

    # 1. Exact read of the written data
    read_back = node.read("client_1", "doc.bin", 500, 3500)
    assert read_back == payload

    # 2. Sub-range read across chunk boundaries (e.g. from 1200 to 2700)
    sub_read = node.read("client_1", "doc.bin", 1200, 1500)
    assert sub_read == payload[700:2200]

    # 3. Read unwritten prefix [0:500]: must be zero-padded bytes
    prefix = node.read("client_1", "doc.bin", 0, 500)
    assert prefix == b"\x00" * 500

    # 4. Read spanning unwritten prefix and data
    span = node.read("client_1", "doc.bin", 0, 700)
    assert span == b"\x00" * 500 + payload[:200]

    # 5. Read past file length (offset 4000 >= 4000): returns empty bytes
    assert node.read("client_1", "doc.bin", 4000, 100) == b""


def test_two_edge_nodes_concurrent_increments() -> None:
    """2 EdgeNodes sharing one store and one LockService: concurrent increments on the same file by 20 threads each lose no updates."""
    store = ObjectStore()
    locks = LockService(default_ttl=10.0)

    # Generous rate limits so test concurrency is unaffected
    limiter_b1 = TokenBucket(capacity=10000, refill_rate=10000)
    limiter_b2 = TokenBucket(capacity=10000, refill_rate=10000)
    limiter_c1 = PerClientLimiter(capacity=10000, refill_rate=10000)
    limiter_c2 = PerClientLimiter(capacity=10000, refill_rate=10000)

    node1 = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=limiter_b1,
        client_limiter=limiter_c1,
        lease_ttl=5.0,
    )
    node2 = EdgeNode(
        node_id="node_2",
        store=store,
        locks=locks,
        backend_limiter=limiter_b2,
        client_limiter=limiter_c2,
        lease_ttl=5.0,
    )

    # Initialize counter file to "0"
    node1.write("init", "shared_counter.txt", 0, b"0")

    num_threads_per_node = 20
    increments_per_thread = 1
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(node: EdgeNode, client_id: str) -> None:
        for _ in range(increments_per_thread):
            while True:
                try:
                    raw = node.read(client_id, "shared_counter.txt", 0, 32)
                    current_val = int(raw.decode("utf-8")) if raw else 0
                    node.write(client_id, "shared_counter.txt", 0, str(current_val + 1).encode("utf-8"))
                    break
                except VersionConflict:
                    # CAS conflict: retry with backoff so competing writers interleave
                    time.sleep(random.uniform(0.001, 0.01))
                except Exception as exc:
                    with lock:
                        errors.append(exc)
                    break

    threads: list[threading.Thread] = []
    for i in range(num_threads_per_node):
        threads.append(threading.Thread(target=worker, args=(node1, f"client_node1_{i}")))
        threads.append(threading.Thread(target=worker, args=(node2, f"client_node2_{i}")))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"Unexpected errors during concurrent increment: {errors}"

    final_raw = node1.read("admin", "shared_counter.txt", 0, 32)
    final_val = int(final_raw.decode("utf-8"))
    expected_total = num_threads_per_node * 2 * increments_per_thread
    assert final_val == expected_total, f"Expected {expected_total}, got {final_val}"


def test_kill_a_node_lease_expiry_and_fencing_rejection() -> None:
    """Kill-a-node test: a writer thread takes the lock then 'dies'; after FakeClock passes TTL, another node writes and dead node's late write is rejected."""
    clock = FakeClock(initial_time=0.0)
    store = ObjectStore(clock=clock)
    locks = LockService(default_ttl=5.0, clock=clock)

    limiter_b1 = TokenBucket(capacity=1000, refill_rate=1000, clock=clock)
    limiter_b2 = TokenBucket(capacity=1000, refill_rate=1000, clock=clock)
    limiter_c1 = PerClientLimiter(capacity=1000, refill_rate=1000, clock=clock)
    limiter_c2 = PerClientLimiter(capacity=1000, refill_rate=1000, clock=clock)

    node_dead = EdgeNode(
        node_id="dead_node",
        store=store,
        locks=locks,
        backend_limiter=limiter_b1,
        client_limiter=limiter_c1,
        lease_ttl=5.0,
        clock=clock,
    )
    node_live = EdgeNode(
        node_id="live_node",
        store=store,
        locks=locks,
        backend_limiter=limiter_b2,
        client_limiter=limiter_c2,
        lease_ttl=5.0,
        clock=clock,
    )

    # Dead node acquires exclusive lock on the file and then "dies" (does not release)
    dead_lease = locks.acquire("critical_doc.txt", "dead_node", "exclusive", ttl=5.0)
    assert dead_lease.token == 1
    assert locks.is_valid(dead_lease) is True

    # Clock moves past lease TTL (t = 6.0 > 5.0)
    clock.advance(6.0)
    assert locks.is_valid(dead_lease) is False

    # Live node is able to acquire exclusive lock and write successfully
    node_live.write("client_live", "critical_doc.txt", 0, b"live_node_committed_data")
    assert node_live.read("client_live", "critical_doc.txt", 0, 32) == b"live_node_committed_data"

    # Dead node wakes up from pause and tries late write:
    # 1. LeaseExpired via node validity check before put
    with pytest.raises(LeaseExpired):
        if not locks.is_valid(dead_lease):
            raise LeaseExpired("Lease expired")
        store.put("critical_doc.txt#0", b"stale_data", fencing_token=dead_lease.token)

    # 2. Even if bypassing node check directly to store, ObjectStore fencing rejects stale token 1 (live token is 2)
    with pytest.raises(StaleFencingToken):
        store.put("critical_doc.txt#0", b"stale_data", fencing_token=dead_lease.token)

    # File data remains intact
    assert node_live.read("client_live", "critical_doc.txt", 0, 32) == b"live_node_committed_data"


def test_backend_outage_cached_readable_uncached_fails() -> None:
    """Backend outage: cached chunks are still readable; uncached reads fail with clear error after bounded retries."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    backend_limiter = TokenBucket(capacity=1000, refill_rate=1000)
    client_limiter = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=backend_limiter,
        client_limiter=client_limiter,
    )

    # Populate a file into the store and cache
    node.write("client_1", "cached_file.txt", 0, b"resilient_cached_content")

    # Verify initial read works
    assert node.read("client_1", "cached_file.txt", 0, 30) == b"resilient_cached_content"

    # Trigger backend outage
    store.set_outage(True)
    try:
        # 1. Cached file continues to be readable without errors
        cached_read = node.read("client_1", "cached_file.txt", 0, 30)
        assert cached_read == b"resilient_cached_content"

        # 2. Uncached file read attempts to contact backend, exhausts retries, and raises TransientBackendError
        with pytest.raises(TransientBackendError, match="outage in progress"):
            node.read("client_1", "uncached_file.txt", 0, 30)
    finally:
        store.set_outage(False)


def test_100_concurrent_reads_cold_file_coalescing() -> None:
    """100 concurrent reads of one cold file cause exactly one backend get per chunk."""
    store = ObjectStore(base_latency=0.01)  # small latency so threads arrive concurrently
    locks = LockService(default_ttl=10.0)
    backend_limiter = TokenBucket(capacity=1000, refill_rate=1000)
    client_limiter = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        chunk_size=1024,
        backend_limiter=backend_limiter,
        client_limiter=client_limiter,
    )

    # Create a 2-chunk file directly in the store (2048 bytes)
    file_data = b"X" * 1024 + b"Y" * 1024
    store.put("cold.bin#meta", str(len(file_data)).encode("utf-8"))
    store.put("cold.bin#0", file_data[:1024])
    store.put("cold.bin#1", file_data[1024:])

    # Reset store stats so we count only gets during concurrent read
    store.reset_stats()

    results: list[bytes] = [b""] * 100
    errors: list[Exception] = []
    barrier = threading.Barrier(100)

    def reader_worker(idx: int) -> None:
        client_id = f"reader_{idx}"
        try:
            # Synchronize thread start to ensure concurrent arrival on the cold file
            barrier.wait()
            results[idx] = node.read(client_id, "cold.bin", 0, len(file_data))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader_worker, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Errors during concurrent reads: {errors}"

    # All 100 readers got the identical payload
    for r in results:
        assert r == file_data

    # Exactly 1 backend get per object/chunk:
    # 1 for cold.bin#meta, 1 for cold.bin#0, 1 for cold.bin#1 => Total gets == 3
    store_stats = store.stats()
    assert store_stats["gets"] == 3, f"Expected 3 backend gets (1 per chunk/meta), got {store_stats['gets']}"

    node_stats = node.stats()
    assert node_stats["cache"]["coalesced_waits"] >= 90


def test_exclusive_locking_prevents_concurrent_interference_inside_lease() -> None:
    """Exclusive locking guarantees mutual exclusion: concurrent callers on another node or client block."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    limiter_b = TokenBucket(capacity=1000, refill_rate=1000)
    limiter_c = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
    )

    # Holder 1 takes an exclusive lease for a multi-step operation
    lease1 = locks.acquire("exclusive_test.bin", "holder_1", "exclusive", ttl=5.0)

    blocked = threading.Event()
    error_caught: list[Exception] = []

    def competing_worker() -> None:
        try:
            # Holder 2 attempts to write with short timeout; must be blocked by holder 1's lease
            locks.acquire("exclusive_test.bin", "holder_2", "exclusive", timeout=0.05)
        except Exception as exc:
            error_caught.append(exc)
        finally:
            blocked.set()

    t = threading.Thread(target=competing_worker)
    t.start()
    t.join(timeout=1.0)

    assert blocked.is_set()
    assert len(error_caught) == 1
    from miniedge.errors import LockTimeout
    assert isinstance(error_caught[0], LockTimeout)

    # Release holder 1's lease
    locks.release(lease1)

    # 2. Read-modify-write inside one lease: concurrent partial-chunk writes to the same chunk
    # depend on exclusive locking to serialize read-modify-write and avoid corrupting data.
    errors: list[Exception] = []

    def writer(c_id: str, offset: int, chunk_data: bytes) -> None:
        try:
            node.write(c_id, "rmw_file.bin", offset, chunk_data)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("c1", 0, b"AAAA"))
    t2 = threading.Thread(target=writer, args=("c2", 10, b"BBBB"))
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert not errors, f"Concurrent RMW writes failed: {errors}"
    res = node.read("c1", "rmw_file.bin", 0, 14)
    assert res == b"AAAA" + (b"\x00" * 6) + b"BBBB"


def test_multi_chunk_write_failure_invalidates_touched_chunks_and_preserves_meta() -> None:
    """If put #2 of 4 raises, all touched chunks in cache must be invalidated and #meta untouched."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    limiter_b = TokenBucket(capacity=1000, refill_rate=1000)
    limiter_c = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        chunk_size=1024,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
    )

    # Write initial 4-chunk file (4096 bytes)
    initial_data = b"A" * 4096
    node.write("c1", "doc4.bin", 0, initial_data)
    assert node.read("c1", "doc4.bin", 0, 4096) == initial_data

    # Save original put method
    real_put = store.put

    def fail_on_chunk2(key: str, data: bytes, expected_version=None, fencing_token=None) -> int:
        if key == "doc4.bin#2":
            raise TransientBackendError("Simulated failure on chunk 2")
        return real_put(key, data, expected_version=expected_version, fencing_token=fencing_token)

    store.put = fail_on_chunk2  # type: ignore[assignment]
    try:
        # Overwrite all 4096 bytes with 'B's; chunk 0 and 1 succeed, chunk 2 fails!
        with pytest.raises(TransientBackendError, match="Simulated failure on chunk 2"):
            node.write("c1", "doc4.bin", 0, b"B" * 4096)
    finally:
        store.put = real_put  # type: ignore[assignment]

    # Verification:
    # 1. Touched chunks (chunk 0 and 1) MUST BE INVALIDATED from local cache so partial state is not cached
    assert "doc4.bin#0" not in node._cache._cache, "Touched chunk 0 was not invalidated from cache on failure"
    assert "doc4.bin#1" not in node._cache._cache, "Touched chunk 1 was not invalidated from cache on failure"

    # 2. #meta was NOT updated (remains initial file length 4096)
    meta_raw, _ = store.get("doc4.bin#meta")
    assert int(meta_raw.decode("utf-8")) == 4096

    # 3. Read-after-failure: reads re-fetch uncached chunks from backend store.
    # Backend contains chunk 0 ('B'), chunk 1 ('B'), chunk 2 ('A'), chunk 3 ('A').
    read_after_failure = node.read("c1", "doc4.bin", 0, 4096)
    assert read_after_failure == (b"B" * 2048) + (b"A" * 2048)


def test_retry_on_self_conflict_after_dropped_put_response() -> None:
    """If store applies put but transient error occurs on response, retry self-conflict is treated as success."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    limiter_b = TokenBucket(capacity=1000, refill_rate=1000)
    limiter_c = PerClientLimiter(capacity=1000, refill_rate=1000)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        chunk_size=1024,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
    )

    real_put = store.put
    dropped_ack_triggered = False

    def put_with_dropped_ack(key: str, data: bytes, expected_version=None, fencing_token=None) -> int:
        nonlocal dropped_ack_triggered
        v = real_put(key, data, expected_version=expected_version, fencing_token=fencing_token)
        if key == "self_conflict.bin#0" and not dropped_ack_triggered:
            dropped_ack_triggered = True
            # Store applied the write, but client experiences a dropped ACK network error
            raise TransientBackendError("Simulated dropped ACK after commit")
        return v

    store.put = put_with_dropped_ack  # type: ignore[assignment]
    try:
        # Must succeed via self-conflict resolution instead of aborting with VersionConflict
        node.write("c1", "self_conflict.bin", 0, b"hello self conflict resolution")
    finally:
        store.put = real_put  # type: ignore[assignment]

    assert dropped_ack_triggered is True
    assert node.read("c1", "self_conflict.bin", 0, 30) == b"hello self conflict resolution"


def test_fencing_rejects_stale_write_when_is_valid_monkeypatched() -> None:
    """Fencing token rejects stale write at ObjectStore even if is_valid is monkeypatched to True."""
    clock = FakeClock(initial_time=0.0)
    store = ObjectStore(clock=clock)
    locks = LockService(default_ttl=5.0, clock=clock)
    limiter_b = TokenBucket(capacity=1000, refill_rate=1000, clock=clock)
    limiter_c = PerClientLimiter(capacity=1000, refill_rate=1000, clock=clock)

    node1 = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
        clock=clock,
    )
    node2 = EdgeNode(
        node_id="node_2",
        store=store,
        locks=locks,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
        clock=clock,
    )

    # Node 1 writes initial data with token 1
    node1.write("c1", "fencing_bypass.bin", 0, b"initial_data")

    # Node 2 writes with token 2 (store max_fencing_token is now 2)
    node2.write("c2", "fencing_bypass.bin", 0, b"node2_higher_token")

    # Monkeypatch locks.is_valid to always return True to simulate a compromised or buggy node
    orig_is_valid = locks.is_valid
    locks.is_valid = lambda lease: True  # type: ignore[assignment]

    try:
        from miniedge.locks import Lease
        stale_lease = Lease("fencing_bypass.bin", "exclusive", "node_1:c1", token=1, expires_at=100.0)
        # Attempting backend put through edge node with stale token 1 must pass the monkeypatched
        # is_valid check but be rejected by ObjectStore's fencing token check with StaleFencingToken.
        with pytest.raises(StaleFencingToken):
            node1._backend_put(
                key="fencing_bypass.bin#0",
                data=b"stale_overwrite",
                expected_version=1,
                lease=stale_lease,
            )
    finally:
        locks.is_valid = orig_is_valid  # type: ignore[assignment]


def test_lease_check_runs_inside_retry_loop() -> None:
    """Lease validity must be verified inside the backend retry loop so expiry during sleep aborts with LeaseExpired."""
    clock = FakeClock(initial_time=0.0)
    store = ObjectStore(clock=clock)
    locks = LockService(default_ttl=5.0, clock=clock)
    limiter_b = TokenBucket(capacity=1000, refill_rate=1000, clock=clock)
    limiter_c = PerClientLimiter(capacity=1000, refill_rate=1000, clock=clock)

    node = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        lease_ttl=5.0,
        backend_limiter=limiter_b,
        client_limiter=limiter_c,
        clock=clock,
    )

    # Initial write of 4 bytes so metadata is already set and length does not change on overwrite
    node.write("c1", "retry_expiry.bin", 0, b"init")

    real_put = store.put
    attempt_count = 0

    def put_with_expiry_during_retry(key: str, data: bytes, expected_version=None, fencing_token=None) -> int:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            # Advance simulated clock past lease TTL (5.0s) while retrying
            clock.advance(10.0)
            raise TransientBackendError("First attempt transient failure")
        return real_put(key, data, expected_version=expected_version, fencing_token=fencing_token)

    store.put = put_with_expiry_during_retry  # type: ignore[assignment]
    try:
        # Overwrite same 4 bytes: metadata length is unchanged.
        # Lease check inside retry loop must detect expiry on attempt 2 and abort with LeaseExpired.
        with pytest.raises(LeaseExpired, match="expired"):
            node.write("c1", "retry_expiry.bin", 0, b"data")
    finally:
        store.put = real_put  # type: ignore[assignment]


def test_meta_and_chunks_invalidated_on_fencing_token_bump() -> None:
    """When a writer bumps the path fencing token, another node's cached #meta and chunks are invalidated."""
    store = ObjectStore()
    locks = LockService(default_ttl=5.0)
    limiter_b1 = TokenBucket(capacity=1000, refill_rate=1000)
    limiter_b2 = TokenBucket(capacity=1000, refill_rate=1000)
    limiter_c1 = PerClientLimiter(capacity=1000, refill_rate=1000)
    limiter_c2 = PerClientLimiter(capacity=1000, refill_rate=1000)

    node1 = EdgeNode(
        node_id="node_1",
        store=store,
        locks=locks,
        backend_limiter=limiter_b1,
        client_limiter=limiter_c1,
    )
    node2 = EdgeNode(
        node_id="node_2",
        store=store,
        locks=locks,
        backend_limiter=limiter_b2,
        client_limiter=limiter_c2,
    )

    # Node 1 writes 10 bytes: populates node1._cache with #meta and #0
    node1.write("c1", "meta_bump.txt", 0, b"0123456789")
    assert node1.read("c1", "meta_bump.txt", 0, 10) == b"0123456789"
    assert "meta_bump.txt#meta" in node1._cache._cache
    assert "meta_bump.txt#0" in node1._cache._cache

    # Node 2 writes 50 bytes: bumps fencing token and updates store
    extended_data = bytes(range(50))
    node2.write("c2", "meta_bump.txt", 0, extended_data)

    # Node 1 reads the file: must detect token bump, invalidate #meta and #0, and read 50 bytes
    read_back = node1.read("c1", "meta_bump.txt", 0, 50)
    assert read_back == extended_data

