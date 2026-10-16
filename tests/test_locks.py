"""Unit, stress, and integration tests for LockService and Lease."""

import random
import threading
import time
import pytest

from miniedge.clock import FakeClock
from miniedge.errors import LeaseExpired, LockTimeout, StaleFencingToken
from miniedge.locks import Lease, LockService
from miniedge.object_store import ObjectStore


def test_compatibility_matrix() -> None:
    """Verify shared-shared concurrency, shared-exclusive blocking, and exclusive-exclusive blocking."""
    service = LockService(default_ttl=5.0)

    # Shared + Shared: both readers acquire concurrently.
    l_r1 = service.acquire("path1", "reader1", "shared")
    l_r2 = service.acquire("path1", "reader2", "shared")
    assert service.is_valid(l_r1) is True
    assert service.is_valid(l_r2) is True

    # Shared + Exclusive: writer is blocked while readers hold shared leases.
    with pytest.raises(LockTimeout):
        service.acquire("path1", "writer1", "exclusive", timeout=0.05)

    # Release both readers.
    service.release(l_r1)
    service.release(l_r2)

    # Exclusive acquires cleanly.
    l_w1 = service.acquire("path1", "writer1", "exclusive")
    assert service.is_valid(l_w1) is True

    # Exclusive + Exclusive: second writer is blocked.
    with pytest.raises(LockTimeout):
        service.acquire("path1", "writer2", "exclusive", timeout=0.05)

    # Exclusive + Shared: new reader is blocked by active exclusive writer.
    with pytest.raises(LockTimeout):
        service.acquire("path1", "reader3", "shared", timeout=0.05)

    service.release(l_w1)

    # Re-entrant acquisition by the same holder on the same path is rejected.
    l_held = service.acquire("path2", "holder1", "shared")
    with pytest.raises(ValueError, match="already holds"):
        service.acquire("path2", "holder1", "shared")
    with pytest.raises(ValueError, match="already holds"):
        service.acquire("path2", "holder1", "exclusive")
    service.release(l_held)


def test_exclusive_token_strictly_increases() -> None:
    """Verify fencing token increases monotonically on exclusive grants and stays constant on shared."""
    service = LockService()

    # First exclusive grant gets token 1.
    l1 = service.acquire("res", "w1", "exclusive")
    assert l1.token == 1
    service.release(l1)

    # Shared grant does not increment fencing token.
    l_s = service.acquire("res", "r1", "shared")
    assert l_s.token == 1
    service.release(l_s)

    # Second exclusive grant gets token 2.
    l2 = service.acquire("res", "w2", "exclusive")
    assert l2.token == 2
    service.release(l2)

    # Third exclusive grant gets token 3.
    l3 = service.acquire("res", "w3", "exclusive")
    assert l3.token == 3
    service.release(l3)


def test_lease_expiry_and_lazy_purge() -> None:
    """Verify expired exclusive lease allows a new holder to acquire without explicit release."""
    clock = FakeClock(initial_time=0.0)
    service = LockService(default_ttl=5.0, clock=clock)

    # Holder A acquires and never releases (simulating crash).
    l_a = service.acquire("file", "holder_A", "exclusive", ttl=5.0)
    assert l_a.token == 1
    assert service.is_valid(l_a) is True

    # Advance clock past TTL.
    clock.advance(6.0)

    # Lease A is now invalid.
    assert service.is_valid(l_a) is False

    # Holder B acquires immediately via lazy purge and receives incremented token.
    l_b = service.acquire("file", "holder_B", "exclusive", ttl=5.0)
    assert l_b.token == 2
    assert l_b.holder_id == "holder_B"
    assert service.is_valid(l_b) is True

    # Releasing an expired or unknown lease is a harmless no-op.
    service.release(l_a)
    assert service.is_valid(l_b) is True


def test_renew_extends_lease_and_expired_raises() -> None:
    """Verify renew() extends expiry and raises LeaseExpired if already expired."""
    clock = FakeClock(initial_time=0.0)
    service = LockService(default_ttl=5.0, clock=clock)

    l1 = service.acquire("doc", "client1", "exclusive", ttl=5.0)
    assert l1.expires_at == 5.0

    clock.advance(3.0)
    # Renew at t=3.0 with ttl=5.0 -> new expiry 8.0.
    l1_renewed = service.renew(l1, ttl=5.0)
    assert l1_renewed.expires_at == 8.0
    assert l1_renewed.token == l1.token
    assert service.is_valid(l1_renewed) is True

    # Advance past renewed expiry (t=9.0 > 8.0).
    clock.advance(6.0)
    assert service.is_valid(l1_renewed) is False

    with pytest.raises(LeaseExpired, match="already expired"):
        service.renew(l1_renewed)


def test_writer_preference() -> None:
    """Verify queued writers take precedence over new readers once active readers exit."""
    service = LockService(default_ttl=10.0)
    events: list[str] = []
    lock = threading.Lock()

    def record(event: str) -> None:
        with lock:
            events.append(event)

    # Reader 1 acquires initial shared lock.
    r1_lease = service.acquire("res", "reader1", "shared")
    record("r1_active")

    writer_ready = threading.Event()
    writer_finished = threading.Event()

    def writer_worker() -> None:
        writer_ready.set()
        w_lease = service.acquire("res", "writer", "exclusive", timeout=2.0)
        record("writer_acquired")
        time.sleep(0.05)
        record("writer_released")
        service.release(w_lease)
        writer_finished.set()

    reader2_finished = threading.Event()

    def reader2_worker() -> None:
        # Wait until writer is blocked in queue before attempting reader 2 acquire.
        writer_ready.wait()
        time.sleep(0.02)
        r2_lease = service.acquire("res", "reader2", "shared", timeout=2.0)
        record("reader2_acquired")
        service.release(r2_lease)
        reader2_finished.set()

    t_writer = threading.Thread(target=writer_worker)
    t_reader2 = threading.Thread(target=reader2_worker)

    t_writer.start()
    t_reader2.start()

    time.sleep(0.08)
    # Release initial reader 1: writer must acquire before reader 2.
    service.release(r1_lease)
    record("r1_released")

    t_writer.join(timeout=3.0)
    t_reader2.join(timeout=3.0)

    # Writer must acquire and release BEFORE reader 2 acquires.
    assert "writer_acquired" in events
    assert "reader2_acquired" in events
    idx_writer_done = events.index("writer_released")
    idx_reader2_got = events.index("reader2_acquired")
    assert idx_writer_done < idx_reader2_got, (
        f"Writer preference violated: events order was {events}"
    )


def test_stress_30_threads_invariant_checker() -> None:
    """Stress test 30 threads across 3 paths asserting mutual exclusion invariants."""
    service = LockService(default_ttl=2.0)
    paths = ["path_alpha", "path_beta", "path_gamma"]
    num_threads = 30
    duration_seconds = 2.0

    # Invariant checker state
    active_exclusive: dict[str, str | None] = {p: None for p in paths}
    active_shared: dict[str, set[str]] = {p: set() for p in paths}
    checker_lock = threading.Lock()
    errors: list[str] = []
    stop_event = threading.Event()

    def worker(worker_id: int) -> None:
        holder_id = f"worker_{worker_id}"
        rnd = random.Random(worker_id)

        while not stop_event.is_set():
            path = rnd.choice(paths)
            mode = "exclusive" if rnd.random() < 0.3 else "shared"

            try:
                lease = service.acquire(path, holder_id, mode, timeout=0.2, ttl=1.0)
            except LockTimeout:
                continue
            except Exception as exc:
                errors.append(f"Unexpected acquire error: {exc}")
                continue

            # Verify and update invariant state under test lock.
            with checker_lock:
                if mode == "exclusive":
                    if active_exclusive[path] is not None:
                        errors.append(
                            f"Exclusive overlap on {path}: {active_exclusive[path]} and {holder_id}"
                        )
                    if len(active_shared[path]) > 0:
                        errors.append(
                            f"Exclusive {holder_id} overlaps active shared {active_shared[path]} on {path}"
                        )
                    active_exclusive[path] = holder_id
                else:
                    if active_exclusive[path] is not None:
                        errors.append(
                            f"Shared {holder_id} overlaps exclusive {active_exclusive[path]} on {path}"
                        )
                    active_shared[path].add(holder_id)

            # Simulated critical section work
            time.sleep(rnd.uniform(0.001, 0.005))

            # Exit critical section
            with checker_lock:
                if mode == "exclusive":
                    active_exclusive[path] = None
                else:
                    active_shared[path].remove(holder_id)

            service.release(lease)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for thread in threads:
        thread.start()

    time.sleep(duration_seconds)
    stop_event.set()

    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors, f"Invariant violations detected during lock stress test: {errors}"


def test_fencing_scenario_with_object_store() -> None:
    """Verify fencing tokens protect ObjectStore from stale writes after lease expiry."""
    clock = FakeClock(initial_time=0.0)
    store = ObjectStore(clock=clock)
    service = LockService(default_ttl=5.0, clock=clock)

    # Node A acquires exclusive lock and gets token 1.
    lease_a = service.acquire("doc_path", "node_A", "exclusive", ttl=5.0)
    assert lease_a.token == 1

    # Node A pauses (simulated by advancing clock past lease TTL).
    clock.advance(10.0)

    # Node B acquires exclusive lock and gets token 2.
    lease_b = service.acquire("doc_path", "node_B", "exclusive", ttl=5.0)
    assert lease_b.token == 2

    # Node B writes to ObjectStore with fencing token 2.
    new_v = store.put("doc_path", b"node_b_data", fencing_token=lease_b.token)
    assert new_v == 1

    # Node A resumes and tries to write with stale fencing token 1; rejected!
    with pytest.raises(StaleFencingToken, match="Stale fencing token 1"):
        store.put("doc_path", b"node_a_stale_data", fencing_token=lease_a.token)

    # ObjectStore retains Node B's valid data.
    data, version = store.get("doc_path")
    assert data == b"node_b_data"
    assert version == 1


def test_writer_timeout_wakes_blocked_readers() -> None:
    """When a waiting writer times out, it must decrement waiting_writers AND notify queued readers."""
    service = LockService(default_ttl=10.0)

    # Reader 1 holds shared lease with long TTL.
    r1 = service.acquire("path_w_timeout", "reader1", "shared", ttl=10.0)

    writer_started = threading.Event()
    writer_failed = threading.Event()

    def writer_worker() -> None:
        writer_started.set()
        try:
            service.acquire("path_w_timeout", "writer", "exclusive", timeout=0.05)
        except LockTimeout:
            writer_failed.set()

    reader2_started = threading.Event()
    reader2_acquired = threading.Event()

    def reader2_worker() -> None:
        # Ensure writer is waiting before reader 2 attempts acquisition.
        writer_started.wait()
        time.sleep(0.01)
        reader2_started.set()
        r2 = service.acquire("path_w_timeout", "reader2", "shared", timeout=1.0)
        reader2_acquired.set()
        service.release(r2)

    t_writer = threading.Thread(target=writer_worker)
    t_reader2 = threading.Thread(target=reader2_worker)

    t_writer.start()
    t_reader2.start()

    # Wait for writer to fail on its timeout.
    assert writer_failed.wait(timeout=1.0) is True

    # Reader 2 must acquire immediately without waiting for r1 to release or for its own timeout.
    assert reader2_acquired.wait(timeout=0.2) is True

    t_writer.join(timeout=1.0)
    t_reader2.join(timeout=1.0)
    service.release(r1)


def test_waiters_wake_when_fake_clock_advances_past_lease_expiry() -> None:
    """Waiters blocked in cond.wait() must wake immediately when FakeClock advances past expiry."""
    clock = FakeClock(initial_time=0.0)
    service = LockService(default_ttl=10.0, clock=clock)

    # Holder A holds exclusive lease expiring at t=10.0.
    l_a = service.acquire("path_clock_advance", "holder_A", "exclusive", ttl=10.0)
    assert l_a.token == 1

    holder_b_waiting = threading.Event()
    holder_b_acquired = threading.Event()

    def holder_b_worker() -> None:
        path_lock = service._get_path_lock("path_clock_advance")
        holder_b_waiting.set()
        l_b = service.acquire("path_clock_advance", "holder_B", "exclusive", timeout=60.0, ttl=5.0)
        holder_b_acquired.set()
        service.release(l_b)

    t_b = threading.Thread(target=holder_b_worker)
    t_b.start()

    holder_b_waiting.wait()
    path_lock = service._get_path_lock("path_clock_advance")
    for _ in range(50):
        with path_lock.cond:
            if path_lock.waiting_writers > 0:
                break
        time.sleep(0.01)

    # Simulated time moves past holder A's lease expiry.
    clock.advance(15.0)

    # Without FakeClock notifying registered conditions, holder B would be stuck for 10 real seconds.
    # With registered conditions, holder B acquires immediately (well under 0.5s real time).
    assert holder_b_acquired.wait(timeout=0.5) is True
    t_b.join(timeout=1.0)


def test_release_only_releases_exact_lease_matching_token() -> None:
    """release(lease) must only release if holder_id AND token match the active lease."""
    clock = FakeClock(initial_time=0.0)
    service = LockService(default_ttl=5.0, clock=clock)

    # Node A acquires exclusive lease (token 1).
    lease_a = service.acquire("path_exact", "node_A", "exclusive", ttl=5.0)
    assert lease_a.token == 1

    # Lease A expires.
    clock.advance(6.0)

    # Node B acquires exclusive lease (token 2).
    lease_b = service.acquire("path_exact", "node_B", "exclusive", ttl=5.0)
    assert lease_b.token == 2
    assert service.is_valid(lease_b) is True

    # Node A calls release with its old expired lease: must not affect Node B.
    service.release(lease_a)
    assert service.is_valid(lease_b) is True

    # Even if someone calls release matching Node B's holder_id and expires_at but with stale token 1,
    # Node B's lock must remain valid.
    stale_token_lease = Lease(
        path="path_exact",
        mode="exclusive",
        holder_id=lease_b.holder_id,
        token=lease_a.token,
        expires_at=lease_b.expires_at,
    )
    service.release(stale_token_lease)
    assert service.is_valid(lease_b) is True

    # Releasing the exact active lease releases the lock.
    service.release(lease_b)
    assert service.is_valid(lease_b) is False


def test_fencing_token_monotonic_across_idle_cleanup() -> None:
    """Fencing tokens must never decrease for a path even if idle PathLock state is cleaned up."""
    clock = FakeClock(initial_time=0.0)
    service = LockService(clock=clock)

    # Acquire exclusive: gets token 1.
    l1 = service.acquire("path_cleanup", "writer1", "exclusive")
    assert l1.token == 1
    service.release(l1)

    # Advance clock and run idle cleanup.
    clock.advance(10.0)
    service.cleanup_idle()

    # The idle path lock object must be evicted from internal path map.
    assert "path_cleanup" not in service._paths

    # Acquire again on the same path: token must be 2, not reset to 1.
    l2 = service.acquire("path_cleanup", "writer2", "exclusive")
    assert l2.token == 2
    service.release(l2)

