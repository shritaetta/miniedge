"""Thread-safe chunk cache with LRU eviction and request coalescing."""

from collections import OrderedDict
from concurrent.futures import Future
import threading
from typing import Any, Callable, NamedTuple

from miniedge.clock import Clock


class CacheEntry(NamedTuple):
    """Entry stored in the chunk cache with version and dirty flag."""

    data: bytes
    version: int
    dirty: bool = False


class ChunkCache:
    """Thread-safe LRU chunk cache with single-flight request coalescing."""

    def __init__(
        self,
        loader: Callable[[Any], tuple[bytes, int]],
        capacity: int,
        clock: Clock | None = None,
        on_evict_dirty: Callable[..., None] | None = None,
    ) -> None:
        """Initialize cache with backend loader callable, entry capacity, and optional hooks."""
        if capacity < 1:
            raise ValueError("Cache capacity must be at least 1")

        self._loader = loader
        self._capacity = int(capacity)
        self._on_evict_dirty = on_evict_dirty

        # Standardize clock injection for time-dependent operations.
        if isinstance(clock, Clock):
            self._clock = clock
        elif callable(clock):
            self._clock = Clock(time_fn=clock)
        else:
            self._clock = Clock()

        self._lock = threading.Lock()
        self._cache: OrderedDict[Any, CacheEntry] = OrderedDict()
        self._in_flight: dict[Any, Future[tuple[bytes, int]]] = {}
        self._generations: dict[Any, int] = {}

        self._stats = {
            "hits": 0,
            "misses": 0,
            "coalesced_waits": 0,
            "backend_loads": 0,
            "evictions": 0,
        }

    def __len__(self) -> int:
        """Return the current number of cached entries."""
        # Invariant: protect atomic read of cache size.
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict[str, int]:
        """Return a snapshot copy of cache operation counters."""
        # Invariant: protect atomic read of metrics dictionary.
        with self._lock:
            return dict(self._stats)

    def _evict_under_lock(self) -> list[tuple[Any, CacheEntry]]:
        """Evict least-recently-used entries until cache size <= capacity."""
        evicted_dirty: list[tuple[Any, CacheEntry]] = []
        while len(self._cache) > self._capacity:
            ev_key, ev_entry = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            if ev_entry.dirty:
                evicted_dirty.append((ev_key, ev_entry))
        return evicted_dirty

    def _notify_evict_dirty(self, key: Any, entry: CacheEntry) -> None:
        """Invoke on_evict_dirty hook outside the lock to avoid holding lock during I/O."""
        if self._on_evict_dirty is not None:
            try:
                self._on_evict_dirty(key, entry.data, entry.version)
            except TypeError:
                self._on_evict_dirty(key, entry)

    def get(self, key: Any) -> tuple[bytes, int]:
        """Retrieve entry for key, coalescing concurrent misses into a single loader call."""
        # Invariant: protect atomic cache lookup, LRU touch, and in-flight leader registration.
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats["hits"] += 1
                entry = self._cache[key]
                return entry.data, entry.version

            self._stats["misses"] += 1

            if key in self._in_flight:
                self._stats["coalesced_waits"] += 1
                future = self._in_flight[key]
                is_leader = False
                initial_generation = 0
            else:
                future = Future()
                self._in_flight[key] = future
                self._stats["backend_loads"] += 1
                is_leader = True
                initial_generation = self._generations.get(key, 0)

                # Re-check cache after registering in_flight to prevent check-then-act races.
                if key in self._cache:
                    self._in_flight.pop(key, None)
                    self._stats["backend_loads"] -= 1
                    self._stats["misses"] -= 1
                    self._stats["hits"] += 1
                    self._cache.move_to_end(key)
                    entry = self._cache[key]
                    return entry.data, entry.version

        if not is_leader:
            # Wait outside the lock so other threads can proceed freely.
            data, version = future.result()
            # Invariant: protect LRU touch after receiving coalesced fetch result.
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
            return data, version

        # As leader, invoke loader outside the lock to prevent blocking concurrent cache access.
        try:
            load_result = self._loader(key)
            load_exc = None
        except BaseException as exc:
            load_result = None
            load_exc = exc
        finally:
            # Invariant: protect atomic in-flight cleanup, generation check, and cache commit.
            with self._lock:
                # Pop in_flight only if it is still this leader's registered future.
                if self._in_flight.get(key) is future:
                    self._in_flight.pop(key, None)

                if load_result is not None:
                    # Guard against key resurrection if invalidate() or put() ran during load.
                    current_generation = self._generations.get(key, 0)
                    if current_generation == initial_generation:
                        existing = self._cache.get(key)
                        can_insert = False
                        if existing is None:
                            can_insert = True
                        elif not existing.dirty and load_result[1] > existing.version:
                            can_insert = True

                        if can_insert:
                            self._cache[key] = CacheEntry(
                                data=bytes(load_result[0]),
                                version=int(load_result[1]),
                                dirty=False,
                            )
                            self._cache.move_to_end(key)
                            evicted = self._evict_under_lock()
                        else:
                            evicted = []
                    else:
                        evicted = []
                else:
                    evicted = []

            # Wake up all waiters outside the lock.
            if load_exc is not None:
                future.set_exception(load_exc)
            else:
                future.set_result(load_result)

            # Flush evicted dirty entries outside the lock.
            for ev_key, ev_entry in evicted:
                self._notify_evict_dirty(ev_key, ev_entry)

        if load_exc is not None:
            raise load_exc

        return load_result

    def put(self, key: Any, data: bytes, version: int, dirty: bool = False) -> None:
        """Insert or update a cache entry, updating LRU order and evicting if over capacity."""
        # Invariant: protect atomic cache insertion, LRU update, generation bump, and capacity bounds.
        with self._lock:
            self._generations[key] = self._generations.get(key, 0) + 1
            self._cache[key] = CacheEntry(data=bytes(data), version=int(version), dirty=bool(dirty))
            self._cache.move_to_end(key)
            evicted = self._evict_under_lock()

        # Flush evicted dirty entries outside the lock.
        for ev_key, ev_entry in evicted:
            self._notify_evict_dirty(ev_key, ev_entry)

    def invalidate(self, key: Any) -> bool:
        """Remove key from cache if present, returning True if removed."""
        # Invariant: protect atomic removal of entry from cache and bump generation.
        with self._lock:
            self._generations[key] = self._generations.get(key, 0) + 1
            entry = self._cache.pop(key, None)

        if entry is not None and entry.dirty:
            self._notify_evict_dirty(key, entry)

        return entry is not None
