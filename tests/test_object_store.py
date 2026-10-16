"""Unit and concurrency tests for ObjectStore."""

import threading
import pytest

from miniedge.clock import FakeClock
from miniedge.errors import (
    StaleFencingToken,
    Throttled,
    TransientBackendError,
    VersionConflict,
)
from miniedge.object_store import ObjectStore


def test_basic_get_put_delete() -> None:
    """Verify standard CRUD operations and version increments."""
    store = ObjectStore()

    # Initial put returns version 1.
    v1 = store.put("chunk_1", b"hello")
    assert v1 == 1

    data, version = store.get("chunk_1")
    assert data == b"hello"
    assert version == 1

    # Overwrite increments version monotonically.
    v2 = store.put("chunk_1", b"world")
    assert v2 == 2

    data, version = store.get("chunk_1")
    assert data == b"world"
    assert version == 2

    # Verify complete ObjectRecord tuple metadata.
    record = store.get_record("chunk_1")
    assert record.data == b"world"
    assert record.version == 2
    assert record.max_fencing_token == 0

    # Delete existing key removes it.
    assert store.delete("chunk_1") is True
    assert store.delete("chunk_1") is False

    with pytest.raises(KeyError):
        store.get("chunk_1")

    # Check operation counters.
    stats = store.stats()
    assert stats["puts"] == 2
    assert stats["gets"] == 3
    assert stats["failures"] == 0
    assert stats["throttled"] == 0


def test_cas_conflict() -> None:
    """Verify compare-and-swap detects conflicting versions."""
    store = ObjectStore()
    v1 = store.put("doc", b"initial")
    assert v1 == 1

    # Matching expected version succeeds.
    v2 = store.put("doc", b"updated", expected_version=1)
    assert v2 == 2

    # Stale expected version raises VersionConflict and leaves data unchanged.
    with pytest.raises(VersionConflict, match="expected 1, found 2"):
        store.put("doc", b"stale_update", expected_version=1)

    data, version = store.get("doc")
    assert data == b"updated"
    assert version == 2

    # CAS on non-existent object expecting version 1 fails.
    with pytest.raises(VersionConflict, match="expected 1, found 0"):
        store.put("new_doc", b"data", expected_version=1)

    # CAS on non-existent object expecting version 0 succeeds.
    v_new = store.put("new_doc", b"data", expected_version=0)
    assert v_new == 1


def test_stale_fencing_token_rejected() -> None:
    """Verify fencing tokens enforce monotonicity and reject stale holders."""
    store = ObjectStore()

    # Initial write with fencing token 10 succeeds.
    store.put("fenced_path", b"payload_1", fencing_token=10)
    record = store.get_record("fenced_path")
    assert record.max_fencing_token == 10

    # Equal token is not stale (< 10 is false), so it is accepted.
    store.put("fenced_path", b"payload_2", fencing_token=10)

    # Advancing fencing token updates highest seen.
    store.put("fenced_path", b"payload_3", fencing_token=20)
    assert store.get_record("fenced_path").max_fencing_token == 20

    # Stale tokens (< 20) are strictly rejected.
    with pytest.raises(StaleFencingToken, match="Stale fencing token 19"):
        store.put("fenced_path", b"stale_payload", fencing_token=19)

    with pytest.raises(StaleFencingToken, match="Stale fencing token 5"):
        store.put("fenced_path", b"stale_payload", fencing_token=5)

    # Store content must remain at the last valid write.
    data, version = store.get("fenced_path")
    assert data == b"payload_3"
    assert version == 3


def test_failure_injection_seeded_rng() -> None:
    """Verify failure rate injection is reproducible when seeded."""
    store_a = ObjectStore(failure_rate=0.4, seed=42)
    store_b = ObjectStore(failure_rate=0.4, seed=42)

    results_a: list[bool] = []
    results_b: list[bool] = []

    for i in range(25):
        key = f"key_{i}"
        val = b"sample"

        try:
            store_a.put(key, val)
            results_a.append(True)
        except TransientBackendError:
            results_a.append(False)

        try:
            store_b.put(key, val)
            results_b.append(True)
        except TransientBackendError:
            results_b.append(False)

    # Identical seeds must produce identical failure sequences.
    assert results_a == results_b
    assert False in results_a
    assert True in results_a
    assert store_a.stats()["failures"] > 0

    # Verify runtime failure rate change.
    store_a.set_failure_rate(1.0)
    with pytest.raises(TransientBackendError):
        store_a.get("any_key")

    store_a.set_failure_rate(0.0)
    store_a.put("guaranteed", b"data")
    assert store_a.get("guaranteed")[0] == b"data"


def test_outage_toggle() -> None:
    """Verify outage toggle rejects all requests and recovers cleanly."""
    store = ObjectStore()
    store.put("item", b"value")

    store.set_outage(True)
    with pytest.raises(TransientBackendError, match="outage in progress"):
        store.get("item")

    with pytest.raises(TransientBackendError, match="outage in progress"):
        store.put("item", b"new_value")

    with pytest.raises(TransientBackendError, match="outage in progress"):
        store.delete("item")

    assert store.stats()["failures"] == 3

    # Clear outage: normal operations must resume without data loss.
    store.set_outage(False)
    data, version = store.get("item")
    assert data == b"value"
    assert version == 1


def test_throttling_sliding_window() -> None:
    """Verify sliding 1s window rejects requests exceeding max_calls_per_second."""
    clock = FakeClock(initial_time=100.0)
    store = ObjectStore(max_calls_per_second=2, clock=clock)

    # First two calls within the 1-second window succeed.
    store.put("k", b"v1")
    store.put("k", b"v2")

    # Third call at same timestamp exceeds 2 calls/sec limit.
    with pytest.raises(Throttled, match="exceeded 2 calls/sec"):
        store.put("k", b"v3")

    assert store.stats()["throttled"] == 1

    # Advance clock beyond 1-second window; quota resets.
    clock.advance(1.1)
    store.put("k", b"v4")
    assert store.get("k")[0] == b"v4"


def test_concurrent_puts_no_lost_version_increments() -> None:
    """Verify 50 concurrent threads increment versions without lost updates."""
    num_threads = 50
    store = ObjectStore()
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            # Synchronize threads so they contend on the same key simultaneously.
            barrier.wait()
            store.put("shared_counter", f"data_{worker_id}".encode("utf-8"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors, f"Unexpected errors during concurrent puts: {errors}"

    # Invariant: exactly 50 put operations, resulting in version 50.
    _, final_version = store.get("shared_counter")
    assert final_version == num_threads
    assert store.stats()["puts"] == num_threads


def test_simulated_latency_releases_lock() -> None:
    """Verify the internal store lock is not held while sleeping for latency."""
    clock = FakeClock(initial_time=0.0)
    store = ObjectStore(base_latency=10.0, clock=clock)
    put_finished = threading.Event()

    def slow_put() -> None:
        store.put("slow_key", b"slow_data")
        put_finished.set()

    thread = threading.Thread(target=slow_put)
    thread.start()

    # Wait until worker has computed latency and is waiting in sleep().
    assert clock.wait_for_sleepers(count=1, timeout=2.0)
    assert not put_finished.is_set()

    # Another operation must acquire the store lock immediately without blocking.
    stats = store.stats()
    assert isinstance(stats, dict)

    # Wake sleeper and verify completion.
    clock.advance(10.0)
    thread.join(timeout=2.0)
    assert put_finished.is_set()
    assert store.get_record("slow_key").data == b"slow_data"
