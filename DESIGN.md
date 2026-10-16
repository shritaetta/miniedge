# MiniEdge: Design Document

> Draft. Sections marked **[YOU]** are decisions you must understand and rewrite in your own words before showing this to anyone. Interviewers will ask about these.

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

- No real networking. Edge nodes are objects/threads in one process, and the object store is in-memory or disk-backed with simulated latency.
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
               | get_chunk / put_chunk (rate limited, retried)
               v
+-----------------------------+
|   ObjectStore (object_store.py)
|   latency, failures, versions
+-----------------------------+
```

**Data model:** files are split into fixed-size chunks (default 64 KB). A chunk is identified by `(path, chunk_index)`. Each object has a `version` (monotonically increasing integer) used as an ETag.

## 4. Concurrency model **[YOU]**

**Choice: `threading` (not `asyncio`).**

- The workload is I/O-bound (simulated network latency), so the GIL is not the bottleneck: threads release the GIL while sleeping/waiting.
- I want to demonstrate real synchronization primitives (`Lock`, `Condition`, `Future`) rather than hide them behind an event loop.
- Tradeoff: threads cost more memory and make races easier to write. An `asyncio` design would scale to more concurrent clients and make coalescing simpler, but would hide the lock-ordering reasoning I want to show.
- Rule: **never mix** `asyncio` and `threading` in this codebase.

*Be ready to answer: "Why not multiprocessing?" (CPU-bound work only; here it would add IPC cost and break shared in-memory cache semantics.)*

## 5. Components

### 5.1 ObjectStore

- API: `get(key) -> (data, version)`, `put(key, data, expected_version=None) -> new_version`, `delete(key)`.
- `put` with `expected_version` is **compare-and-swap**: raises `VersionConflict` if the stored version differs.
- Injectable behavior: latency (base + jitter), random failure rate, throttling (raises `Throttled` when called too fast), full outage toggle.
- Thread-safe internally.

### 5.2 ChunkCache (LRU + coalescing)

- Storage: `OrderedDict[key] -> CacheEntry(data, version, dirty)` guarded by one `Lock`; capacity in bytes or entries.
- **Hit path:** acquire lock, look up, `move_to_end`, release, return.
- **Miss path with coalescing:**
  1. Under the lock, check the cache again, then check the `in_flight: dict[key, Future]` map.
  2. If an in-flight fetch exists, release the lock and wait on that Future.
  3. If not, create the Future, register it, **release the lock**, then perform the backend fetch.
  4. After the fetch: re-acquire the lock, insert into the cache, remove from `in_flight`, release, then set the Future's result (or exception).
- **Invariants:**
  - The cache lock is **never held during backend I/O**.
  - The `in_flight` entry is **always removed**, including when the fetch fails (use `try/finally`), so failures don't poison later requests.
  - Exactly one backend fetch per key per miss wave.
- Eviction: LRU. If the evicted entry is dirty (write-back mode), it must be flushed first.

**[YOU] Why LRU?** Simple and good for temporal locality. Weakness: a big sequential scan flushes the whole cache (scan pollution). Alternatives: ARC, 2Q, LFU. State why LRU is acceptable here.

### 5.3 RateLimiter

- **Token bucket:** capacity `C`, refill rate `R` tokens/sec. `try_acquire(n)` is non-blocking; `acquire(n, timeout)` blocks.
- Refill is computed lazily from `time.monotonic()` on each call (no background thread), under a lock.
- Two uses: (a) a **backend limiter** shared by all cache misses on an edge node, (b) **per-client limiters** at the edge entrance.
- **Retry policy** when the backend returns `Throttled` or a transient error: exponential backoff with full jitter, `sleep = random(0, min(cap, base * 2**attempt))`, bounded attempts, then surface the error to the caller.

**[YOU] Why a token bucket over a fixed window or leaky bucket?** Allows controlled bursts and has no boundary-spike problem of fixed windows. Be ready to compare against sliding window log/counter.

### 5.4 LockService (leases + fencing)

- Per-path lock with modes **shared (read)** and **exclusive (write)**.
- Acquiring returns a `Lease(path, mode, holder_id, token, expires_at)`.
- `token` is a **fencing token**: a monotonically increasing integer per path, incremented on every exclusive grant.
- Leases expire after `ttl` seconds unless renewed (`renew(lease)`). An expired lease is treated as released, so a crashed holder never blocks others forever.
- **Fairness:** waiting writers block new readers (prevents writer starvation). Implemented with a `Condition` and a wait queue.
- **Fencing at the store:** every exclusive write passes `fencing_token` to the ObjectStore, which rejects any write whose token is lower than the highest token it has seen for that path. This protects against the classic failure: node A pauses (GC/network), its lease expires, node B acquires and writes, A wakes up and writes stale data.

**[YOU] Lease duration tradeoff:** too short means spurious expiry during slow writes (needs renewal); too long means slow recovery after a crash. State your default and why.

### 5.5 EdgeNode

Read path:
1. Per-client rate limit check.
2. Acquire shared lock (or skip for relaxed reads; decide and document).
3. Validate path fencing token: if token bumped since last visit, invalidate `#meta` and all cached chunks for path.
4. For each chunk in range: cache lookup, coalesced backend fetch on miss. Record observed chunk/meta versions in `_client_read_versions[(client_id, key)]`.
5. Release lock, return bytes.

Write path (**write-through first**):
1. Per-client rate limit check.
2. Acquire exclusive lock, get fencing token.
3. Validate path fencing token: if token bumped, invalidate cached `#meta` and chunks.
4. Read-modify-write affected chunks; pop `expected_version` from `_client_read_versions` and `put` to the object store with `fencing_token` and `expected_version`. Track all touched chunk keys.
5. If file length changed or was newly created, write `#meta` **after** all chunk writes succeed.
6. Update local cache with newly committed versions. On any error, invalidate all touched keys and `#meta` from cache. Release lock.

**Optimistic Concurrency & Client Read-Version Map:**
- **Key:** `(client_id, object_key)` tuple, uniquely tracking what version of each chunk or metadata object a specific client observed.
- **Lifetime:** Populated during `read()` when a client reads `#meta` or individual chunk keys (`path#chunk_idx`). Retained while the client processes the data locally.
- **Cleanup:** Popped on `write()` when committing modified chunks back to the object store via CAS (`expected_version`). If unwritten, entries persist until overwritten by future reads or cleared.
- **Path-level fencing invalidation:** Edge nodes track the latest fencing token seen per path in `_path_fencing_tokens`. When acquiring a lease, if `lease.token > last_seen_token`, all local cache entries for `#meta` and associated chunks are invalidated, ensuring multi-node consistency without cache snooping.

Optional extension: **write-back** (write to the cache, mark dirty, flush asynchronously or on eviction/`fsync`).

**[YOU] Write-through vs write-back:**

| | Write-through | Write-back |
|---|---|---|
| Latency | Backend latency on every write | Local speed |
| Durability | Durable on return | Data loss if node crashes before flush |
| Complexity | Low | Dirty tracking, flush ordering, crash recovery |

**Write-Back Implementation & Flusher Mechanics:**
- **Write Path:** When `write_back=True`, `write()` modifies the in-memory `ChunkCache` and marks chunk and metadata entries `dirty=True`. It captures the original version in `_dirty_chunks` for eventual CAS validation. Crucially, it returns immediately **without touching the backend** (no store calls, no backend rate limiter token consumption).
- **Background Flusher & Close:** A dedicated background thread periodically wakes up every `flush_interval` (using the injectable clock) to call `flush()`. Calling `close()` signals the flusher thread to terminate, joins the thread, and executes a final synchronous `flush()`.
- **Flush Ordering & Fencing:** `flush(path)` acquires an exclusive lease from `LockService` to obtain a fresh `fencing_token`. It commits dirty chunk payloads to `ObjectStore` first via CAS (`expected_version`) and fencing. Only after all chunks successfully commit does it commit the updated file length to `#meta`. Once verified on backend, cached entries transition to clean (`dirty=False`).
- **Dirty Eviction Pre-Flush:** If cache capacity is reached, `ChunkCache` invokes `on_evict_dirty` before evicting a dirty entry. The node acquires an exclusive lease and synchronously writes the evicted chunk to `ObjectStore`, guaranteeing dirty data is never dropped on LRU churn.

**What is lost if the node crashes before a flush:**
- **Volatile Dirty Data Loss:** Any dirty chunk payload and metadata modifications currently held only in volatile memory (`_cache` and `_dirty_chunks`) that have not yet been written via `flush()`, background flusher, or LRU eviction hook are **permanently lost**.
- **Backend Consistency:** The `ObjectStore` retains the exact state from the last successful flush. Other nodes or a newly spawned edge node reading from `ObjectStore` will observe old/unmodified data (or missing file records if never previously flushed).
- **File Length Integrity:** Because `#meta` is strictly written to `ObjectStore` *after* all constituent chunks have been written, a crash never leaves `#meta` pointing past data that failed to reach the store. The file boundary remains consistent with persisted chunks.

## 6. Failure scenarios and expected behavior **[YOU]**

| Scenario | Expected behavior | Mechanism |
|---|---|---|
| 100 threads read the same cold chunk | 1 backend fetch, 100 correct results | Coalescing |
| Backend fetch fails during coalesced read | All waiters get the error; next request retries cleanly | `finally` cleanup of `in_flight` |
| Backend throttles | Callers back off and retry; no retry storm | Token bucket + jittered backoff |
| Writer holds lock and dies | Others acquire after TTL | Lease expiry |
| Paused writer resumes after losing lock | Its write is rejected | Fencing token |
| Two nodes write same chunk without locks | One gets `VersionConflict` | CAS on version |
| Backend fully down | Reads of cached data still succeed; misses fail fast with a clear error | Cache + bounded retries |
| Reader/writer starvation | Writers not starved by continuous readers | Writer-preference queue |

## 7. Testing strategy

- **Unit:** LRU eviction order, token refill math (with an injectable clock), lock mode compatibility matrix.
- **Race/stress:** 100+ threads on one key; mixed readers/writers on one path; verify invariants (backend call count, no lost updates, no two exclusive holders at once).
- **Failure injection:** random backend errors, lease expiry mid-write, simulated pause between lock grant and write.
- **Determinism:** inject a fake clock and seeded RNG so timing-based tests aren't flaky.
- Run stress tests repeatedly (e.g. `pytest --count=50`) to surface rare interleavings.

## 8. Benchmarks

Report, with graphs:
- Throughput and p50/p95/p99 latency vs. number of concurrent clients, with and without cache.
- Cache hit rate vs. cache size for a Zipfian access pattern.
- Backend calls saved by coalescing.
- Effect of rate limiting on error rate under overload.

Fill in real numbers only from your own runs.

## 9. Known limitations **[YOU]**

- Single-process simulation; no real network partitions.
- Lock service is a single point of failure.
- Lease expiry relies on `time.monotonic()` within one process; real systems must handle clock skew between machines.
- Fixed-size chunking with no deduplication or compression.
- LRU is vulnerable to scan pollution.
- **Multi-chunk writes are not atomic:** Puts are applied chunk-by-chunk to the object store. If a write spanning multiple chunks fails midway (e.g. put #2 of 4 raises, process crash, or lease expiry), the chunks written prior to failure remain committed in the backend store, while subsequent chunks and `#meta` are not written. Readers after failure observe partially committed files unless an external transaction or manifest log rollback is implemented.
- **Retries can outlast the TTL:** Exponential backoff retries in `retry_with_backoff` during backend operations may sleep longer than the remaining lease TTL. If the lease expires while retrying, subsequent puts are rejected by fencing or the lease check. `EdgeNode` checks `locks.is_valid(lease)` inside the retry loop before each put attempt to fail fast with `LeaseExpired`.
- **Per-client state growth:** The client read-version map (`_client_read_versions`) and per-client rate limiters (`PerClientLimiter`) retain state keyed by `(client_id, key)` and `client_id`. Clients that read without writing, or clients that disconnect without cleanup, lead to unbounded dictionary memory growth unless an explicit LRU/TTL eviction or session-pruning mechanism is added.

## 10. Future work

ARC/2Q eviction, write-back with a write-ahead log, read-ahead for sequential access, multi-process nodes over sockets, a Raft-backed lock service.
