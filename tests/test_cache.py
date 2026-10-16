"""Unit and concurrency tests for ChunkCache."""

import random
import threading
import time
import pytest

from miniedge.cache import ChunkCache
from miniedge.errors import TransientBackendError


def test_lru_eviction_order() -> None:
    """Verify least-recently-used eviction order and on_evict_dirty callback."""
    evicted_dirty_calls: list[tuple[str, bytes, int]] = []

    def on_evict_dirty(key: str, data: bytes, version: int) -> None:
        evicted_dirty_calls.append((key, data, version))

    def dummy_loader(key: str) -> tuple[bytes, int]:
        return f"loaded_{key}".encode(), 1

    cache = ChunkCache(loader=dummy_loader, capacity=3, on_evict_dirty=on_evict_dirty)

    # Insert 3 keys.
    cache.put("k1", b"v1", 1)
    cache.put("k2", b"v2", 1)
    cache.put("k3", b"v3", 1)
    assert len(cache) == 3

    # Access k1 so that k2 becomes the least recently used key.
    assert cache.get("k1") == (b"v1", 1)

    # Insert k4; capacity 3 exceeded, k2 must be evicted.
    cache.put("k4", b"v4", 1)
    assert len(cache) == 3

    # Stats: k1 was a hit; checking k1, k3, k4 must be hits without calling loader.
    stats_before = cache.stats()
    assert cache.get("k1") == (b"v1", 1)
    assert cache.get("k3") == (b"v3", 1)
    assert cache.get("k4") == (b"v4", 1)
    stats_after = cache.stats()
    assert stats_after["hits"] == stats_before["hits"] + 3
    assert stats_after["backend_loads"] == 0

    # k2 was evicted, so get("k2") triggers loader.
    assert cache.get("k2") == (b"loaded_k2", 1)
    assert cache.stats()["backend_loads"] == 1

    # Verify dirty entry eviction triggers callback.
    dirty_cache = ChunkCache(loader=dummy_loader, capacity=2, on_evict_dirty=on_evict_dirty)
    dirty_cache.put("d1", b"dirty_data", 5, dirty=True)
    dirty_cache.put("c1", b"clean_data", 5, dirty=False)
    dirty_cache.put("d2", b"dirty_data_2", 5, dirty=True)  # Evicts d1

    assert len(evicted_dirty_calls) == 1
    assert evicted_dirty_calls[0] == ("d1", b"dirty_data", 5)

    # Invalidate dirty entry also triggers callback.
    dirty_cache.invalidate("d2")
    assert len(evicted_dirty_calls) == 2
    assert evicted_dirty_calls[1] == ("d2", b"dirty_data_2", 5)


def test_100_threads_coalescing() -> None:
    """Verify 100 concurrent get() calls on a cold key trigger exactly one backend load."""
    num_threads = 100
    barrier = threading.Barrier(num_threads)
    loader_calls = 0
    lock = threading.Lock()

    def slow_loader(key: str) -> tuple[bytes, int]:
        nonlocal loader_calls
        with lock:
            loader_calls += 1
        # Simulate slow I/O transit.
        time.sleep(0.05)
        return b"shared_data", 10

    cache = ChunkCache(loader=slow_loader, capacity=10)
    results: list[tuple[bytes, int]] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()
            res = cache.get("cold_key")
            with lock:
                results.append(res)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors, f"Unexpected errors in coalescing test: {errors}"
    assert len(results) == num_threads
    assert all(r == (b"shared_data", 10) for r in results)
    assert loader_calls == 1

    stats = cache.stats()
    assert stats["backend_loads"] == 1
    assert stats["coalesced_waits"] == num_threads - 1
    assert stats["misses"] == num_threads
    assert stats["hits"] == 0


def test_loader_raises_propagates_and_retries() -> None:
    """Verify loader exceptions wake all waiters with the error and clear in_flight."""
    num_waiters = 10
    barrier = threading.Barrier(num_waiters)
    loader_calls = 0
    should_fail = True
    lock = threading.Lock()

    def failing_loader(key: str) -> tuple[bytes, int]:
        nonlocal loader_calls
        with lock:
            loader_calls += 1
        time.sleep(0.05)
        if should_fail:
            raise TransientBackendError("Backend unavailable")
        return b"recovered_data", 20

    cache = ChunkCache(loader=failing_loader, capacity=10)
    caught_errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()
            cache.get("error_key")
        except TransientBackendError as exc:
            with lock:
                caught_errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_waiters)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    # All waiters must catch the exact exception.
    assert len(caught_errors) == num_waiters
    assert loader_calls == 1

    # in_flight must be cleared so failure does not poison future requests.
    assert len(cache._in_flight) == 0

    # Next get() retries cleanly.
    should_fail = False
    result = cache.get("error_key")
    assert result == (b"recovered_data", 20)
    assert loader_calls == 2


def test_50_threads_mixed_workload_stress() -> None:
    """Verify concurrent get/put/invalidate operations maintain capacity bounds without errors."""
    num_threads = 50
    duration_seconds = 2.0
    capacity = 10
    num_keys = 20
    keys = [f"key_{i}" for i in range(num_keys)]

    def loader(key: str) -> tuple[bytes, int]:
        return f"data_for_{key}".encode(), 1

    cache = ChunkCache(loader=loader, capacity=capacity)
    errors: list[Exception] = []
    stop_event = threading.Event()

    def worker() -> None:
        rnd = random.Random()
        try:
            while not stop_event.is_set():
                key = rnd.choice(keys)
                action = rnd.randint(0, 2)
                if action == 0:
                    cache.get(key)
                elif action == 1:
                    cache.put(key, b"new_val", 2, dirty=(rnd.random() < 0.2))
                else:
                    cache.invalidate(key)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    time.sleep(duration_seconds)
    stop_event.set()

    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors, f"Errors encountered during mixed workload: {errors}"
    assert len(cache) <= capacity


def test_different_keys_load_in_parallel() -> None:
    """Verify distinct cold keys load concurrently without serializing on the cache lock."""
    def slow_loader(key: str) -> tuple[bytes, int]:
        # Latency of 0.2s for each distinct key.
        time.sleep(0.2)
        return f"content_{key}".encode(), 1

    cache = ChunkCache(loader=slow_loader, capacity=10)
    barrier = threading.Barrier(2)
    results: dict[str, tuple[bytes, int]] = {}
    lock = threading.Lock()

    def worker(key: str) -> None:
        barrier.wait()
        res = cache.get(key)
        with lock:
            results[key] = res

    start_time = time.monotonic()
    t1 = threading.Thread(target=worker, args=("key_a",))
    t2 = threading.Thread(target=worker, args=("key_b",))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    elapsed = time.monotonic() - start_time

    assert results["key_a"] == (b"content_key_a", 1)
    assert results["key_b"] == (b"content_key_b", 1)

    # Serial execution would take >= 0.40s. Parallel execution takes ~0.20s (well below 0.35s).
    assert elapsed < 0.35, f"Expected parallel load in ~0.20s, but elapsed time was {elapsed:.3f}s"


def test_regression_get_racing_invalidate_prevents_resurrection() -> None:
    """Verify invalidate() during an in-flight load prevents resurrecting the stale key."""
    load_started = threading.Event()
    allow_finish = threading.Event()

    def gated_loader(key: str) -> tuple[bytes, int]:
        load_started.set()
        allow_finish.wait()
        return b"stale_loaded_data", 1

    cache = ChunkCache(loader=gated_loader, capacity=10)
    result_holder: list[tuple[bytes, int]] = []

    def get_worker() -> None:
        result_holder.append(cache.get("resurrect_key"))

    t = threading.Thread(target=get_worker)
    t.start()

    # Wait until loader has started and is gated inside loader.
    load_started.wait()

    # Invalidate the key while load is in-flight.
    cache.invalidate("resurrect_key")

    # Unblock loader to finish.
    allow_finish.set()
    t.join(timeout=5.0)

    # Loader returns data to the original caller...
    assert result_holder[0] == (b"stale_loaded_data", 1)

    # ...but the key must NOT be resurrected in the cache.
    assert "resurrect_key" not in cache._cache
    assert len(cache) == 0


def test_regression_get_racing_dirty_put_preserves_dirty_entry() -> None:
    """Verify in-flight load does not overwrite concurrent dirty put of the same or newer version."""
    load_started = threading.Event()
    allow_finish = threading.Event()

    def gated_loader(key: str) -> tuple[bytes, int]:
        load_started.set()
        allow_finish.wait()
        return b"stale_loaded_data", 1

    cache = ChunkCache(loader=gated_loader, capacity=10)

    t = threading.Thread(target=lambda: cache.get("dirty_key"))
    t.start()

    # Wait until loader has started.
    load_started.wait()

    # Concurrent dirty put with same version 1.
    cache.put("dirty_key", b"dirty_val", version=1, dirty=True)

    # Allow loader to complete.
    allow_finish.set()
    t.join(timeout=5.0)

    # Dirty entry must survive untouched in the cache.
    entry = cache._cache["dirty_key"]
    assert entry.dirty is True
    assert entry.data == b"dirty_val"
    assert entry.version == 1


def test_regression_leader_raises_base_exception_wakes_waiters() -> None:
    """Verify that if the leader raises a BaseException, waiters do not hang and in_flight is empty."""
    load_started = threading.Event()
    allow_finish = threading.Event()

    class CustomSystemExit(BaseException):
        pass

    def gated_loader(key: str) -> tuple[bytes, int]:
        load_started.set()
        allow_finish.wait()
        raise CustomSystemExit("fatal abort")

    cache = ChunkCache(loader=gated_loader, capacity=10)
    leader_error: list[BaseException] = []
    waiter_error: list[BaseException] = []

    def leader_worker() -> None:
        try:
            cache.get("fatal_key")
        except BaseException as exc:
            leader_error.append(exc)

    def waiter_worker() -> None:
        try:
            cache.get("fatal_key")
        except BaseException as exc:
            waiter_error.append(exc)

    t_leader = threading.Thread(target=leader_worker)
    t_waiter = threading.Thread(target=waiter_worker)

    t_leader.start()
    load_started.wait()

    # Start waiter; it will coalesce on the in-flight future.
    t_waiter.start()
    # Wait without sleep until waiter has entered coalesced wait.
    while cache.stats()["coalesced_waits"] < 1:
        pass

    # Let leader finish and raise BaseException.
    allow_finish.set()

    t_leader.join(timeout=5.0)
    t_waiter.join(timeout=5.0)

    assert not t_leader.is_alive(), "Leader hung"
    assert not t_waiter.is_alive(), "Waiter hung"

    assert len(leader_error) == 1 and isinstance(leader_error[0], CustomSystemExit)
    assert len(waiter_error) == 1 and isinstance(waiter_error[0], CustomSystemExit)
    assert len(cache._in_flight) == 0
