# MiniEdge: Design Document

## 1. Goal

MiniEdge simulates a Nasuni-style architecture on one machine: **edge nodes** serve file reads/writes to clients, keep hot data in a local cache, and talk to a slow, rate-limited, occasionally failing **object store** that holds the source of truth. Multiple edge nodes may touch the same file, so writes are coordinated with **leased locks**.

Concurrency problems this project demonstrates:

| Problem | Component |
|---|---|
| Many clients, same hot data | Thread-safe LRU cache |
| Thundering herd on a cache miss | Request coalescing |
| Protecting the backend from overload | Token bucket + retry with backoff |
| Two nodes editing the same file | Read/write leases + fencing tokens |
| Node crash mid-write | Lease expiry + fencing |
| Conflicting writes | Versioning (ETag-style) |

## 2. Non-goals

- No real networking. Edge nodes are objects/threads in one process, and the object store is an in-memory dictionary with simulated latency, throttling, and failures. Nothing is persisted.
- No distributed consensus. The lock service is a single in-process authority (a real one would need Raft/Paxos or a managed service).
- No POSIX/SMB/NFS protocol support.

## 3. Architecture

```
 clients (threads)
     |  read(path, offset, n) / write(path, offset, data)
     v
+-----------------------------+
|        EdgeNode (edge.py)   |
|  - per-client RateLimiter   |
|  - LockClient  --------------------> LockService (locks.py)
|  - ChunkCache (cache.py)    |
|  - backend RateLimiter      |
+--------------|--------------+
               | get / put (rate limited, retried)
               v
+-----------------------------+
|   ObjectStore (object_store.py)
|   latency, failures, versions
+-----------------------------+
```

**Data model:** files are split into fixed-size chunks (default 64 KB). Chunk *i* of a file at `path` is stored as an object under the key `path#<i>` (for example `doc.bin#0`, `doc.bin#1`). The file length is stored in a separate metadata object under `path#meta`, as an ASCII-encoded integer. Each object has a `version` (monotonically increasing integer) used as an ETag. Chunks that were never written do not exist in the store; a missing key is treated as empty, and reads zero-pad up to the length recorded in `#meta`.

## 4. Concurrency model

**Choice: `threading` (not `asyncio`).**

- The workload is I/O-bound (simulated network latency), and threads release the GIL while sleeping or waiting, so backend-bound paths (for example cache-off reads) scale with thread count. Cache-hit reads, by contrast, are CPU-bound Python and do contend on the GIL, which shows up in the benchmarks as throughput falling as threads increase.
- I want to demonstrate real synchronization primitives (`Lock`, `Condition`, `Future`) rather than hide them behind an event loop.
- Tradeoff: threads cost more memory and make races easier to write. An `asyncio` design would scale to more concurrent clients and make coalescing simpler, but would hide the lock-ordering reasoning I want to show.
- Multiprocessing is not used: the work is not CPU-bound, and it would add IPC cost and break shared in-memory cache semantics.
- Rule: **never mix** `asyncio` and `threading` in this codebase.
- Time is injected through a `Clock` abstraction so tests can use a `FakeClock` (see section 7).

## 5. Components

### 5.1 ObjectStore

- API: `get(key) -> (data, version)`, `get_record(key) -> ObjectRecord(data, version, max_fencing_token)`, `put(key, data, expected_version=None, fencing_token=None) -> new_version`, `delete(key)`.
- `put` with `expected_version` is **compare-and-swap**: raises `VersionConflict` if the stored version differs. A missing object counts as version 0.
- `put` with `fencing_token` rejects the write with `StaleFencingToken` if the token is lower than the highest token the store has seen **for that object key** (for example `doc.bin#0`, `doc.bin#meta`).
- Injectable behavior: latency (base + jitter), random failure rate, throttling (raises `Throttled` when more than N calls arrive in a 1-second sliding window), full outage toggle.
- The simulated latency sleep happens outside the internal lock, so concurrent requests overlap instead of serializing.
- Failure injection uses a seeded RNG so failure sequences are reproducible.
- Thread-safe internally.

### 5.2 ChunkCache (LRU + coalescing)

- Storage: `OrderedDict[key] -> CacheEntry(data, version, dirty)` guarded by one `Lock`; capacity in entries.
- **Hit path:** acquire lock, look up, `move_to_end`, release, return.
- **Miss path with coalescing:**
  1. Under the lock, check the cache again, then check the `in_flight: dict[key, Future]` map.
  2. If an in-flight fetch exists, release the lock and wait on that Future.
  3. If not, create the Future, register it, **release the lock**, then perform the backend fetch.
  4. After the fetch: re-acquire the lock, insert into the cache, remove from `in_flight`, release, then set the Future's result (or exception).
- **Invariants:**
  - The cache lock is **never held during backend I/O**.
  - The `in_flight` entry is **always removed**, including when the fetch fails (use `try/finally`, catching `BaseException`), so failures don't poison later requests.
  - Exactly one backend fetch per key per miss wave.
- **Generation counters:** each key has a generation number that `put()` and `invalidate()` increment. A leader records the generation before loading and only inserts its result if the generation is unchanged, and only if it would not overwrite a dirty or newer entry. This prevents a stale in-flight load from resurrecting an invalidated key or overwriting a concurrent dirty put. Waiters and the leader still receive the loaded data; it just is not cached.
- Eviction: LRU. If the evicted entry is dirty (write-back mode), it is flushed through the `on_evict_dirty` hook, which runs outside the cache lock.

**Why LRU?** It is simple, O(1) with `OrderedDict`, and good for temporal locality. Its weakness is scan pollution (one large sequential read flushes the hot set); ARC or 2Q would address that and are listed under future work.

### 5.3 RateLimiter

- **Token bucket:** capacity `C`, refill rate `R` tokens/sec. `try_acquire(n)` is non-blocking; `acquire(n, timeout)` blocks.
- Refill is computed lazily from the injectable `Clock` (a wrapper over `time.monotonic()` in production) on each call, under a lock, with no background thread.
- `acquire` uses a virtual reservation timeline (`_next_available`): a caller reserves its slot, releases the lock, and sleeps outside the lock, so later callers queue behind it in FIFO order. A call that cannot be satisfied within its timeout returns False without reserving, so it does not delay anyone else. `try_acquire` cannot steal tokens pledged to sleeping waiters.
- Two uses: (a) a **backend limiter** shared by all cache misses and puts on an edge node, which is **blocking** (`acquire`) and therefore smooths load, and (b) **per-client limiters** at the edge entrance, which are **non-blocking** (`try_acquire`) and raise `RateLimited` when a client is over its limit.
- **Sizing rule:** if the backend allows N calls in any 1-second window, choose `C + R * 1s <= N` so the limiter can never release more than N calls in a window.
- **Retry policy** when the backend returns `Throttled` or a transient error: exponential backoff with full jitter, `sleep = random(0, min(cap, base * 2**attempt))`, bounded attempts (default 5 retries), then surface the error to the caller.

**Why a token bucket?** It allows controlled bursts and has no boundary-spike problem of fixed windows. A leaky bucket smooths output but does not allow bursts, and a sliding-window log is exact but costs O(N) memory.

### 5.4 LockService (leases + fencing)

- Per-path lock with modes **shared (read)** and **exclusive (write)**. Each path has its own `Lock` and `Condition`, so contention on one path does not affect others.
- Acquiring returns a `Lease(path, mode, holder_id, token, expires_at)`. Re-entrant acquisition by the same holder on the same path is rejected.
- `token` is a **fencing token**: a monotonically increasing integer per path, incremented on every exclusive grant. Shared leases carry the current token without incrementing it. Tokens survive idle cleanup of path state, so they never reset.
- Leases expire after `ttl` seconds (default 5 s) unless renewed (`renew(lease)`). Expiry is **lazy**: there is no reaper thread. Every operation purges expired leases first, and waiters sleep until the earliest lease expiry or their own deadline, so a crashed holder blocks others for at most one TTL.
- `release(lease)` only takes effect if holder, token, and expiry all match the active lease, so an expired holder cannot release a newer holder's lock.
- **Fairness:** waiting writers block new readers (prevents writer starvation). This is implemented with a `waiting_writers` counter and a `Condition`. A writer that times out decrements the counter and notifies waiters so queued readers are not blocked unnecessarily.
- **Fencing at the store:** every exclusive write passes `fencing_token` to the ObjectStore, which rejects any write whose token is lower than the highest token it has seen for that object key. This protects against the classic failure: node A pauses (GC/network), its lease expires, node B acquires and writes, A wakes up and writes stale data. A pre-write `is_valid()` check alone is not enough because the check and the write are not atomic.

**Lease duration tradeoff:** too short causes spurious expiry during slow writes; too long slows recovery after a crash. 5 s is the simulator default.

### 5.5 EdgeNode

Read path:
1. Per-client rate limit check.
2. Acquire a shared lease (always; this keeps writers out mid-read and gives the fencing token used in step 3).
3. Validate path fencing token: if the token is higher than the last one this node saw for the path, invalidate `#meta` and all cached chunks for the path.
4. Read `#meta` for the file length and clamp the read. For each chunk in range: cache lookup, coalesced backend fetch on miss. Record observed chunk/meta versions in `_client_read_versions[(client_id, key)]`.
5. Release lock, return bytes (zero-padding holes).

Write path (**write-through first**):
1. Per-client rate limit check.
2. Acquire exclusive lock, get fencing token.
3. Validate path fencing token: if token bumped, invalidate cached `#meta` and chunks.
4. Read-modify-write affected chunks; pop `expected_version` from `_client_read_versions` and `put` to the object store with `fencing_token` and `expected_version`. Track all touched chunk keys.
5. If file length changed or was newly created, write `#meta` **after** all chunk writes succeed.
6. Update local cache with newly committed versions. On any error, invalidate all touched keys and `#meta` from cache. Release lock.

**Backend puts:** every put passes the backend rate limiter, then `retry_with_backoff`. The lease is re-validated inside the retry loop before each attempt, so a lease that expires during backoff aborts with `LeaseExpired`. If a put was applied but its response was lost (dropped ACK), the retry sees a `VersionConflict` against its own write; the node detects this by checking the stored record (same data, `version == expected + 1`, same token) and treats it as success.

**Optimistic Concurrency & Client Read-Version Map:**
- **Key:** `(client_id, object_key)` tuple, tracking what version of each chunk or metadata object a specific client observed.
- **Lifetime:** Populated during `read()` when a client reads `#meta` or individual chunk keys (`path#chunk_idx`). Retained while the client processes the data locally.
- **Cleanup:** Popped on `write()` when committing modified chunks back to the object store via CAS (`expected_version`). If unwritten, entries persist until overwritten by future reads or cleared.
- **Path-level fencing invalidation:** Edge nodes track the latest fencing token seen per path in `_path_tokens`. When acquiring a lease, if `lease.token > last_seen_token`, all local cache entries for `#meta` and associated chunks are invalidated, ensuring multi-node consistency without cache snooping or invalidation messages.

Optional extension: **write-back** (write to the cache, mark dirty, flush asynchronously or on eviction/`flush()`).

| | Write-through | Write-back |
|---|---|---|
| Latency | Backend latency on every write | Local speed |
| Durability | Durable on return | Data loss if node crashes before flush |
| Complexity | Low | Dirty tracking, flush ordering, crash recovery |

**Write-Back Implementation & Flusher Mechanics:**
- **Write Path:** When `write_back=True`, `write()` modifies the in-memory `ChunkCache` and marks chunk and metadata entries `dirty=True`. It captures the original version in `_dirty_chunks` for eventual CAS validation. Crucially, it returns immediately **without touching the backend** (no store calls, no backend rate limiter token consumption, and no lease acquisition).
- **Background Flusher & Close:** A dedicated background thread periodically wakes up every `flush_interval` (using the injectable clock) to call `flush()`. Calling `close()` signals the flusher thread to terminate, joins the thread, and executes a final synchronous `flush()`.
- **Flush Ordering & Fencing:** `flush(path)` acquires an exclusive lease from `LockService` to obtain a fresh `fencing_token`. It commits dirty chunk payloads to `ObjectStore` first via CAS (`expected_version`) and fencing. Only after all chunks successfully commit does it commit the updated file length to `#meta`. Once verified on backend, cached entries transition to clean (`dirty=False`).
- **Dirty Eviction Pre-Flush:** If cache capacity is reached, `ChunkCache` invokes `on_evict_dirty` before evicting a dirty entry. The node acquires an exclusive lease and synchronously writes the evicted chunk to `ObjectStore`, guaranteeing dirty data is never dropped on LRU churn.

**What is lost if the node crashes before a flush:**
- **Volatile Dirty Data Loss:** Any dirty chunk payload and metadata modifications currently held only in volatile memory (`_cache` and `_dirty_chunks`) that have not yet been written via `flush()`, background flusher, or LRU eviction hook are **permanently lost**.
- **Backend Consistency:** The `ObjectStore` retains the exact state from the last successful flush. Other nodes or a newly spawned edge node reading from `ObjectStore` will observe old/unmodified data (or missing file records if never previously flushed).
- **File Length Integrity:** Because `#meta` is strictly written to `ObjectStore` *after* all constituent chunks have been written, a crash never leaves `#meta` pointing past data that failed to reach the store. The file boundary remains consistent with persisted chunks.

## 6. Failure scenarios and expected behavior

| Scenario | Expected behavior | Mechanism |
|---|---|---|
| 100 threads read the same cold chunk | 1 backend fetch, 100 correct results | Coalescing |
| Backend fetch fails during coalesced read | All waiters get the error; next request retries cleanly | `finally` cleanup of `in_flight` |
| Backend throttles | Callers back off and retry; no retry storm | Token bucket + jittered backoff |
| Writer holds lock and dies | Others acquire after TTL | Lease expiry |
| Paused writer resumes after losing lock | Its write is rejected | Fencing token |
| Two nodes write same chunk without locks | One gets `VersionConflict` | CAS on version |
| Backend fully down | Reads of cached data still succeed; misses fail after bounded retries with a clear `TransientBackendError` | Cache + bounded retries |
| Reader/writer starvation | Writers not starved by continuous readers | Writer-preference via `waiting_writers` |
| Write response lost after store applied it | Retry is treated as success, not a conflict | Self-conflict check in `_backend_put` |
| Lease expires during retry backoff | Write aborts with `LeaseExpired` | Lease check inside the retry loop |

## 7. Testing strategy

- **Unit:** LRU eviction order, token refill math (with an injectable clock), lock mode compatibility matrix, CAS and fencing behavior in the store.
- **Race/stress:** 100+ threads on one key; mixed readers/writers on one path with an invariant checker that flags any two exclusive holders or any shared/exclusive overlap; a multi-node chaos test (2 nodes, 40 threads, 10 files, 10% injected backend failures) checked against a ground-truth model, with a join timeout to detect deadlocks.
- **Failure injection:** random backend errors, backend outage and recovery, lease expiry mid-write using a deliberately slow store, dropped-ACK puts, partial multi-chunk write failure, throttling with the limiter on vs off.
- **Regression tests:** a stale in-flight load must not resurrect an invalidated key or overwrite a dirty put; a leader raising `BaseException` must wake waiters; a timed-out writer must wake queued readers; fencing tokens must stay monotonic across idle cleanup.
- **Determinism:** a `FakeClock` (`advance()`, blocking `sleep()`, `wait_for_sleepers()`) lets lease-expiry, refill, and flusher tests run instantly and deterministically; the lock service registers its conditions with the clock so waiters wake when simulated time advances. Backend failures use seeded RNGs, and test seeds are printed and can be replayed with `MINIEDGE_TEST_SEED`.
- The chaos test is marked `slow` and can be skipped with `pytest -m "not slow"`.

## 8. Benchmarks

Report, with graphs:
- Throughput and p50/p95/p99 latency vs. number of concurrent clients, with and without cache.
- Cache hit rate vs. cache size for a Zipfian access pattern.
- Backend calls saved by coalescing.
- Effect of rate limiting on error rate under overload.

**How to read the numbers:** this is a simulation. The 20 ms backend latency is a parameter (`time.sleep`), not a measurement of any real object store, and all numbers come from running the real code on real threads on one machine. They are valid for comparing mechanisms against each other, not as production performance claims. Specifically:
- The cache-off run bypasses the cache so every read hits the simulated backend; a read costs two serial backend calls (`#meta` plus the chunk), hence roughly 41 ms.
- The "without coalescing" backend-call count in the coalescing experiment is the analytic expectation (one call per concurrent miss); the measured result is that coalescing issues exactly one call. A stress test confirms this end to end through `EdgeNode` using the store's own call counter.
- The rate-limiter experiment offers a burst of N simultaneous requests, not a sustained N req/s. Eliminating errors is bought with latency: goodput with the limiter is bounded by the refill rate divided by backend calls per read.

## 9. Known limitations

- Single-process simulation; no real network partitions.
- Lock service is a single point of failure.
- Lease expiry relies on the monotonic clock within one process; real systems must handle clock skew between machines.
- Fixed-size chunking with no deduplication or compression.
- LRU is vulnerable to scan pollution.
- **Multi-chunk writes are not atomic:** Puts are applied chunk-by-chunk to the object store. If a write spanning multiple chunks fails midway (e.g. put #2 of 4 raises, process crash, or lease expiry), the chunks written prior to failure remain committed in the backend store, while subsequent chunks and `#meta` are not written. Readers after failure observe partially committed files unless an external transaction or manifest log rollback is implemented.
- **Retries can outlast the TTL:** Exponential backoff retries in `retry_with_backoff` during backend operations may sleep longer than the remaining lease TTL. If the lease expires while retrying, subsequent puts are rejected by fencing or the lease check. `EdgeNode` checks `locks.is_valid(lease)` inside the retry loop before each put attempt to fail fast with `LeaseExpired`.
- **Unbounded per-client and per-key state:** The client read-version map (`_client_read_versions`), per-client rate limiters (`PerClientLimiter`), and the cache's per-key `_generations` map retain state that is never pruned. Clients that read without writing, or disconnect without cleanup, cause unbounded dictionary growth unless LRU/TTL eviction or session pruning is added.
- **Write-back skips locking on the write path:** two nodes dirtying the same file are only reconciled at flush time, where CAS and fencing catch the conflict.
- **Self-invalidation:** every exclusive grant bumps the path's fencing token, so a node's own sequential writes invalidate its cached chunks for that path on its next access.
- **Token-bucket reservations are not rolled back:** if a thread is aborted while sleeping on a reserved slot, the reservation stays consumed.



ARC/2Q eviction, write-back with a write-ahead log, a manifest or WAL for atomic multi-chunk writes, lease renewal during long writes, read-ahead for sequential access, multi-process nodes over sockets, a Raft-backed lock service.
