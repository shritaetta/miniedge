"""EdgeNode: client-facing caching node coordinating leases, coalesced chunks, and rate limits."""

import random
import threading
from typing import Any

from miniedge.cache import CacheEntry, ChunkCache
from miniedge.clock import Clock, FakeClock
from miniedge.errors import (
    LeaseExpired,
    RateLimited,
    VersionConflict,
)
from miniedge.locks import Lease, LockService
from miniedge.object_store import ObjectStore
from miniedge.ratelimit import PerClientLimiter, TokenBucket, retry_with_backoff


class EdgeNode:
    """Client-facing edge caching filesystem node.

    Coordinates client access to a distributed object store using:
    - Ingress per-client rate limiting.
    - Path-level leased distributed locking via LockService.
    - Fixed-size chunking and single-flight request coalescing via ChunkCache.
    - Egress backend rate limiting and jittered backoff retries.
    """

    def __init__(
        self,
        node_id: str,
        store: ObjectStore,
        locks: LockService,
        *,
        chunk_size: int = 64 * 1024,
        cache_capacity: int = 256,
        backend_limiter: TokenBucket,
        client_limiter: PerClientLimiter,
        lease_ttl: float = 5.0,
        clock: Clock | None = None,
        write_back: bool = False,
        flush_interval: float = 0.5,
    ) -> None:
        """Initialize EdgeNode with backend store, lock service, and operational limiters."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if cache_capacity <= 0:
            raise ValueError("cache_capacity must be positive")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")

        self._node_id = str(node_id)
        self._store = store
        self._locks = locks
        self._chunk_size = int(chunk_size)
        self._cache_capacity = int(cache_capacity)
        self._backend_limiter = backend_limiter
        self._client_limiter = client_limiter
        self._lease_ttl = float(lease_ttl)
        self._write_back = bool(write_back)
        self._flush_interval = float(flush_interval)

        if isinstance(clock, Clock):
            self._clock = clock
        elif callable(clock):
            self._clock = Clock(time_fn=clock)
        else:
            self._clock = Clock()

        self._rng = random.Random()
        self._lock = threading.RLock()

        # Operation and throughput metrics
        self._reads = 0
        self._writes = 0
        self._bytes_read = 0
        self._bytes_written = 0

        # Track versions observed during read() by (client_id, key) for CAS expected_version
        self._client_read_versions: dict[tuple[str, str], int] = {}
        # Cache coherence: track last seen fencing token per path to invalidate stale cache across nodes
        self._path_tokens: dict[str, int] = {}
        self._path_keys: dict[str, set[str]] = {}

        # Write-back dirty chunk tracking: maps key -> original expected_version for backend CAS
        self._dirty_chunks: dict[str, int] = {}
        self._dirty_paths: set[str] = set()
        self._stop_flusher = threading.Event()
        self._flusher_cond = threading.Condition()
        self._flusher_thread: threading.Thread | None = None

        if self._write_back:
            self._clock.register_condition(self._flusher_cond)

        # Local chunk cache with single-flight request coalescing and dirty eviction hook
        self._cache = ChunkCache(
            loader=self._backend_loader,
            capacity=self._cache_capacity,
            clock=self._clock,
            on_evict_dirty=self._on_cache_evict_dirty if self._write_back else None,
        )

        if self._write_back and self._flush_interval > 0:
            self._flusher_thread = threading.Thread(
                target=self._flusher_loop,
                name=f"{self._node_id}_flusher",
                daemon=True,
            )
            self._flusher_thread.start()

    def _flusher_loop(self) -> None:
        """Periodic background flusher committing dirty cache chunks to ObjectStore."""
        deadline = self._clock.now() + self._flush_interval
        with self._flusher_cond:
            while not self._stop_flusher.is_set():
                now = self._clock.now()
                if now >= deadline:
                    deadline = now + self._flush_interval
                    self._flusher_cond.release()
                    try:
                        self.flush()
                    except Exception:
                        # Transient errors during background flush will be retried on next interval
                        pass
                    finally:
                        self._flusher_cond.acquire()
                else:
                    remaining = deadline - now
                    timeout = max(0.01, remaining) if not isinstance(self._clock, FakeClock) else None
                    self._flusher_cond.wait(timeout=timeout)

    def _on_cache_evict_dirty(self, key: str, data: bytes, version: int) -> None:
        """Flush evicted dirty chunk to ObjectStore immediately so data is not lost."""
        path = key.rsplit("#", 1)[0]
        with self._lock:
            expected_ver = self._dirty_chunks.pop(key, version)

        holder_id = f"{self._node_id}:evict_flusher"
        lease = self._locks.acquire(path, holder_id, "exclusive", ttl=self._lease_ttl)
        try:
            self._backend_put(
                key=key,
                data=data,
                expected_version=expected_ver,
                lease=lease,
            )
        finally:
            self._locks.release(lease)
            with self._lock:
                if not any(k.rsplit("#", 1)[0] == path for k in self._dirty_chunks):
                    self._dirty_paths.discard(path)

    def _track_path_key(self, path: str, key: str) -> None:
        """Associate a cache key with a file path for token-based invalidation."""
        with self._lock:
            keys = self._path_keys.get(path)
            if keys is None:
                keys = set()
                self._path_keys[path] = keys
            keys.add(key)

    def _validate_path_cache(self, path: str, current_token: int) -> None:
        """Invalidate cached chunks for path if an external writer bumped the fencing token."""
        with self._lock:
            last_token = self._path_tokens.get(path)
            if last_token is not None and current_token > last_token:
                # An exclusive lease was granted to another writer; cached chunks for this path are stale.
                keys_to_invalidate = list(self._path_keys.get(path, set()))
                for k in keys_to_invalidate:
                    self._cache.invalidate(k)
                self._path_keys[path] = set()

            self._path_tokens[path] = current_token

    def _backend_loader(self, key: str) -> tuple[bytes, int]:
        """Fetch chunk or metadata from ObjectStore under backend rate limiting and retry."""
        # Backend rate limit check prevents overwhelming slow or metered cloud storage.
        if not self._backend_limiter.acquire(1):
            raise RateLimited("Backend rate limiter rejected request")

        def fetch() -> tuple[bytes, int]:
            try:
                return self._store.get(key)
            except KeyError:
                # Missing chunk or metadata in backend represents unwritten / empty data.
                return (b"", 0)

        return retry_with_backoff(
            fetch,
            clock=self._clock,
            rng=self._rng,
        )

    def _backend_put(
        self,
        key: str,
        data: bytes,
        expected_version: int,
        lease: Lease,
    ) -> int:
        """Store chunk or metadata payload with CAS validation, fencing protection, and lease checks."""
        # Rate limit egress puts to the backend store.
        if not self._backend_limiter.acquire(1):
            raise RateLimited("Backend rate limiter rejected request")

        def do_put() -> int:
            # Lease validity must be checked inside the retry loop before each put attempt
            if not self._locks.is_valid(lease):
                raise LeaseExpired(f"Lease for '{lease.path}' expired during retry backoff")
            return self._store.put(
                key,
                data,
                expected_version=expected_version,
                fencing_token=lease.token,
            )

        try:
            return retry_with_backoff(
                do_put,
                clock=self._clock,
                rng=self._rng,
            )
        except VersionConflict:
            # Check if a previous attempt in this retry wave succeeded on the store before a network ACK failure
            try:
                rec = self._store.get_record(key)
                if (
                    rec.data == data
                    and rec.version == expected_version + 1
                    and rec.max_fencing_token == lease.token
                ):
                    return rec.version
            except Exception:
                pass

            # Stale local cache entry caused a version mismatch; invalidate so next read reloads fresh state.
            self._cache.invalidate(key)
            raise

    def read(self, client_id: str, path: str, offset: int, length: int) -> bytes:
        """Read length bytes from path at offset under a shared lease, coalescing chunk misses."""
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        if length == 0:
            return b""

        # Ingress client rate limit enforcement
        if not self._client_limiter.allow(client_id):
            raise RateLimited(f"Client '{client_id}' rate limit exceeded")

        # Shared lock guarantees no concurrent writer is actively modifying chunks during the read.
        holder_id = f"{self._node_id}:{client_id}"
        lease = self._locks.acquire(path, holder_id, "shared", ttl=self._lease_ttl)
        try:
            # Check if an external writer bumped the fencing token since our last access
            self._validate_path_cache(path, lease.token)

            # File length is tracked in a dedicated metadata object stored at f"{path}#meta".
            meta_key = f"{path}#meta"
            self._track_path_key(path, meta_key)
            meta_raw, meta_version = self._cache.get(meta_key)
            with self._lock:
                self._client_read_versions[(client_id, meta_key)] = meta_version
            file_length = int(meta_raw.decode("utf-8")) if meta_raw else 0

            # Reads starting at or beyond the current file boundary return empty bytes.
            if offset >= file_length:
                return b""

            # Clamp read length to existing file size.
            actual_read_len = min(length, file_length - offset)
            if actual_read_len <= 0:
                return b""

            start_chunk = offset // self._chunk_size
            end_chunk = (offset + actual_read_len - 1) // self._chunk_size

            collected_chunks: list[bytes] = []
            for chunk_idx in range(start_chunk, end_chunk + 1):
                chunk_key = f"{path}#{chunk_idx}"
                self._track_path_key(path, chunk_key)
                chunk_raw, chunk_version = self._cache.get(chunk_key)
                with self._lock:
                    self._client_read_versions[(client_id, chunk_key)] = chunk_version

                chunk_start = chunk_idx * self._chunk_size
                chunk_end = chunk_start + self._chunk_size

                slice_start = max(offset, chunk_start) - chunk_start
                slice_end = min(offset + actual_read_len, chunk_end) - chunk_start

                # Missing chunks or sparse holes read as zero bytes up to the file's recorded length.
                if len(chunk_raw) < slice_end:
                    chunk_raw = chunk_raw + (b"\x00" * (slice_end - len(chunk_raw)))

                collected_chunks.append(chunk_raw[slice_start:slice_end])

            result = b"".join(collected_chunks)
            with self._lock:
                self._reads += 1
                self._bytes_read += len(result)
            return result
        finally:
            self._locks.release(lease)

    def write(self, client_id: str, path: str, offset: int, data: bytes) -> None:
        """Write data to path at offset under an exclusive lease with write-through or write-back commits."""
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")

        # Ingress client rate limit enforcement
        if not self._client_limiter.allow(client_id):
            raise RateLimited(f"Client '{client_id}' rate limit exceeded")

        if self._write_back:
            self._write_back_update(client_id, path, offset, data)
            return

        # Exclusive lock serializes writes and provides fencing token for ObjectStore verification.
        holder_id = f"{self._node_id}:{client_id}"
        lease = self._locks.acquire(path, holder_id, "exclusive", ttl=self._lease_ttl)
        touched_keys: list[str] = []
        meta_key = f"{path}#meta"
        try:
            # Check if an external writer bumped the fencing token since our last access
            self._validate_path_cache(path, lease.token)

            self._track_path_key(path, meta_key)
            meta_raw, meta_version = self._cache.get(meta_key)
            curr_file_len = int(meta_raw.decode("utf-8")) if meta_raw else 0
            new_file_len = max(curr_file_len, offset + len(data))

            if len(data) > 0:
                start_chunk = offset // self._chunk_size
                end_chunk = (offset + len(data) - 1) // self._chunk_size

                for chunk_idx in range(start_chunk, end_chunk + 1):
                    chunk_key = f"{path}#{chunk_idx}"
                    touched_keys.append(chunk_key)
                    self._track_path_key(path, chunk_key)
                    chunk_start = chunk_idx * self._chunk_size
                    chunk_end = chunk_start + self._chunk_size

                    write_start = max(offset, chunk_start)
                    write_end = min(offset + len(data), chunk_end)

                    in_chunk_offset = write_start - chunk_start
                    data_slice = data[write_start - offset : write_end - offset]

                    # Read-modify-write: fetch current chunk data and version
                    existing_data, existing_version = self._cache.get(chunk_key)

                    # Expand buffer if writing past existing chunk length
                    buf = bytearray(existing_data)
                    needed_len = in_chunk_offset + len(data_slice)
                    if len(buf) < needed_len:
                        buf.extend(b"\x00" * (needed_len - len(buf)))
                    buf[in_chunk_offset : in_chunk_offset + len(data_slice)] = data_slice
                    new_chunk_data = bytes(buf)

                    with self._lock:
                        expected_version = self._client_read_versions.pop(
                            (client_id, chunk_key), existing_version
                        )

                    # Write-through: commit chunk to ObjectStore with fencing token and CAS check
                    new_version = self._backend_put(
                        key=chunk_key,
                        data=new_chunk_data,
                        expected_version=expected_version,
                        lease=lease,
                    )

                    # Update local cache with newly committed version
                    self._cache.put(chunk_key, new_chunk_data, new_version, dirty=False)

            # Update file length metadata if file expanded or was just created (#meta written AFTER chunks)
            if new_file_len != curr_file_len or meta_version == 0:
                new_meta_data = str(new_file_len).encode("utf-8")

                with self._lock:
                    expected_meta_version = self._client_read_versions.pop(
                        (client_id, meta_key), meta_version
                    )

                new_meta_version = self._backend_put(
                    key=meta_key,
                    data=new_meta_data,
                    expected_version=expected_meta_version,
                    lease=lease,
                )
                self._cache.put(meta_key, new_meta_data, new_meta_version, dirty=False)

            with self._lock:
                self._writes += 1
                self._bytes_written += len(data)
        except Exception:
            # If any chunk put raises or lease expires midway, invalidate all touched chunks and #meta
            # so the local cache does not retain partial, uncommitted state.
            for k in touched_keys:
                self._cache.invalidate(k)
            self._cache.invalidate(meta_key)
            raise
        finally:
            self._locks.release(lease)

    def stats(self) -> dict[str, Any]:
        """Return merged operation and performance metrics from cache, limiter, and counters."""
        with self._lock:
            base_metrics: dict[str, Any] = {
                "node_id": self._node_id,
                "reads": self._reads,
                "writes": self._writes,
                "bytes_read": self._bytes_read,
                "bytes_written": self._bytes_written,
                "backend_limiter_tokens": self._backend_limiter.tokens,
            }
        cache_metrics = self._cache.stats()
        return {**cache_metrics, **base_metrics, "cache": cache_metrics}

    def _write_back_update(
        self,
        client_id: str,
        path: str,
        offset: int,
        data: bytes,
    ) -> None:
        """Update in-memory chunk cache and mark entries dirty without touching backend."""
        meta_key = f"{path}#meta"
        to_cache: list[tuple[str, bytes, int]] = []

        with self._lock:
            # File length is determined from cached metadata or previously observed client read version
            with self._cache._lock:
                meta_entry = self._cache._cache.get(meta_key)
            if meta_entry is not None:
                curr_file_len = int(meta_entry.data.decode("utf-8")) if meta_entry.data else 0
                meta_ver = meta_entry.version
            else:
                curr_file_len = 0
                meta_ver = self._client_read_versions.pop((client_id, meta_key), 0)

            new_file_len = max(curr_file_len, offset + len(data))

            if meta_key not in self._dirty_chunks:
                self._dirty_chunks[meta_key] = meta_ver

            new_meta_data = str(new_file_len).encode("utf-8")
            to_cache.append((meta_key, new_meta_data, meta_ver))
            self._dirty_paths.add(path)

            if len(data) > 0:
                start_chunk = offset // self._chunk_size
                end_chunk = (offset + len(data) - 1) // self._chunk_size

                for chunk_idx in range(start_chunk, end_chunk + 1):
                    chunk_key = f"{path}#{chunk_idx}"
                    chunk_start = chunk_idx * self._chunk_size
                    chunk_end = chunk_start + self._chunk_size

                    write_start = max(offset, chunk_start)
                    write_end = min(offset + len(data), chunk_end)

                    in_chunk_offset = write_start - chunk_start
                    data_slice = data[write_start - offset : write_end - offset]

                    with self._cache._lock:
                        chunk_entry = self._cache._cache.get(chunk_key)

                    if chunk_entry is not None:
                        existing_data = chunk_entry.data
                        existing_ver = chunk_entry.version
                    else:
                        existing_data = b""
                        existing_ver = self._client_read_versions.pop((client_id, chunk_key), 0)

                    if chunk_key not in self._dirty_chunks:
                        self._dirty_chunks[chunk_key] = existing_ver

                    buf = bytearray(existing_data)
                    needed_len = in_chunk_offset + len(data_slice)
                    if len(buf) < needed_len:
                        buf.extend(b"\x00" * (needed_len - len(buf)))
                    buf[in_chunk_offset : in_chunk_offset + len(data_slice)] = data_slice
                    new_chunk_data = bytes(buf)

                    to_cache.append((chunk_key, new_chunk_data, existing_ver))
                    self._dirty_paths.add(path)

            self._writes += 1
            self._bytes_written += len(data)

        # Update cache outside self._lock so eviction hooks can perform backend I/O without holding self._lock
        for k, k_data, k_ver in to_cache:
            self._cache.put(k, k_data, k_ver, dirty=True)
            self._track_path_key(path, k)

    def _flush_path(self, path: str) -> None:
        """Flush dirty chunks and metadata for a specific path to ObjectStore."""
        with self._lock:
            dirty_keys = [k for k in self._dirty_chunks if k.rsplit("#", 1)[0] == path]
            if not dirty_keys:
                self._dirty_paths.discard(path)
                return

        meta_key = f"{path}#meta"
        chunk_keys = [k for k in dirty_keys if k != meta_key]

        def get_chunk_idx(k: str) -> int:
            try:
                return int(k.rsplit("#", 1)[1])
            except (ValueError, IndexError):
                return 0

        chunk_keys.sort(key=get_chunk_idx)
        has_dirty_meta = meta_key in dirty_keys

        holder_id = f"{self._node_id}:flusher"
        lease = self._locks.acquire(path, holder_id, "exclusive", ttl=self._lease_ttl)
        try:
            self._validate_path_cache(path, lease.token)

            # 1. Flush chunks first before metadata so file length never exposes uncommitted data
            for chunk_key in chunk_keys:
                with self._lock:
                    expected_version = self._dirty_chunks.get(chunk_key)
                    if expected_version is None:
                        continue
                    with self._cache._lock:
                        entry = self._cache._cache.get(chunk_key)
                    if entry is None:
                        self._dirty_chunks.pop(chunk_key, None)
                        continue
                    data_to_write = entry.data

                new_version = self._backend_put(
                    key=chunk_key,
                    data=data_to_write,
                    expected_version=expected_version,
                    lease=lease,
                )

                with self._lock:
                    with self._cache._lock:
                        curr_entry = self._cache._cache.get(chunk_key)
                        if curr_entry is not None and curr_entry.data == data_to_write:
                            self._cache._cache[chunk_key] = CacheEntry(
                                data=data_to_write,
                                version=new_version,
                                dirty=False,
                            )
                            self._dirty_chunks.pop(chunk_key, None)
                        else:
                            self._dirty_chunks[chunk_key] = new_version

            # 2. Flush metadata after chunks succeed
            if has_dirty_meta:
                with self._lock:
                    meta_expected_version = self._dirty_chunks.get(meta_key)
                    meta_data_to_write = None
                    if meta_expected_version is not None:
                        with self._cache._lock:
                            meta_entry = self._cache._cache.get(meta_key)
                        if meta_entry is not None:
                            meta_data_to_write = meta_entry.data
                        else:
                            self._dirty_chunks.pop(meta_key, None)

                if meta_data_to_write is not None:
                    new_meta_version = self._backend_put(
                        key=meta_key,
                        data=meta_data_to_write,
                        expected_version=meta_expected_version,
                        lease=lease,
                    )
                    with self._lock:
                        with self._cache._lock:
                            curr_meta = self._cache._cache.get(meta_key)
                            if curr_meta is not None and curr_meta.data == meta_data_to_write:
                                self._cache._cache[meta_key] = CacheEntry(
                                    data=meta_data_to_write,
                                    version=new_meta_version,
                                    dirty=False,
                                )
                                self._dirty_chunks.pop(meta_key, None)
                            else:
                                self._dirty_chunks[meta_key] = new_meta_version
        finally:
            self._locks.release(lease)
            with self._lock:
                if not any(k.rsplit("#", 1)[0] == path for k in self._dirty_chunks):
                    self._dirty_paths.discard(path)

    def flush(self, path: str | None = None) -> None:
        """Flush dirty chunks and metadata to ObjectStore for a path or all dirty paths."""
        if path is not None:
            self._flush_path(path)
        else:
            with self._lock:
                paths = list(self._dirty_paths)
            for p in paths:
                self._flush_path(p)

    def close(self) -> None:
        """Stop background flusher thread and flush all dirty chunks."""
        self._stop_flusher.set()
        with self._flusher_cond:
            self._flusher_cond.notify_all()
        if self._flusher_thread is not None and self._flusher_thread.is_alive():
            if threading.current_thread() != self._flusher_thread:
                self._flusher_thread.join(timeout=5.0)
            self._flusher_thread = None
        self._clock.unregister_condition(self._flusher_cond)
        self.flush()

    def __enter__(self) -> "EdgeNode":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
