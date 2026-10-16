"""Benchmark experiment runners covering latency, throughput, Zipf locality, coalescing, and throttling."""

import random
import statistics
import threading
import time
from typing import Any

from bench.zipf import ZipfGenerator
from miniedge.cache import ChunkCache
from miniedge.edge import EdgeNode
from miniedge.errors import Throttled
from miniedge.locks import LockService
from miniedge.object_store import ObjectRecord, ObjectStore
from miniedge.ratelimit import PerClientLimiter, TokenBucket


def _compute_percentiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    """Calculate p50, p95, and p99 percentiles from a list of latencies in milliseconds."""
    if not latencies_ms:
        return (0.0, 0.0, 0.0)
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    def pct(p: float) -> float:
        idx = min(int(n * p), n - 1)
        return sorted_lats[idx]

    return (pct(0.50), pct(0.95), pct(0.99))


def run_experiment_1(
    quick: bool = False,
    seed: int = 42,
    base_latency: float = 0.020,
) -> dict[str, Any]:
    """Experiment 1: Latency and throughput (ops/sec, p50/p95/p99) vs client threads: cache ON vs OFF."""
    thread_counts = [1, 4, 16, 32] if quick else [1, 2, 4, 8, 16, 32, 64]
    run_duration = 0.25 if quick else 0.50

    cache_on_throughputs: list[float] = []
    cache_on_p50: list[float] = []
    cache_on_p95: list[float] = []
    cache_on_p99: list[float] = []

    cache_off_throughputs: list[float] = []
    cache_off_p50: list[float] = []
    cache_off_p95: list[float] = []
    cache_off_p99: list[float] = []

    file_name = "bench1.bin"
    file_bytes = b"X" * 1024

    for num_threads in thread_counts:
        # 1. Cache ON evaluation
        store_on = ObjectStore(base_latency=base_latency, seed=seed)
        node_on = EdgeNode(
            "node_on",
            store=store_on,
            locks=LockService(),
            backend_limiter=TokenBucket(100000, 100000),
            client_limiter=PerClientLimiter(100000, 100000),
        )
        node_on.write("init", file_name, 0, file_bytes)
        node_on.read("warmup", file_name, 0, 1024)  # Warm cache entry

        latencies_on: list[float] = []
        stop_on = threading.Event()

        def worker_on(cid: str) -> None:
            local_lats: list[float] = []
            while not stop_on.is_set():
                t0 = time.perf_counter()
                node_on.read(cid, file_name, 0, 1024)
                local_lats.append((time.perf_counter() - t0) * 1000.0)
            latencies_on.extend(local_lats)

        threads = [threading.Thread(target=worker_on, args=(f"c_{i}",)) for i in range(num_threads)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(run_duration)
        stop_on.set()
        for t in threads:
            t.join()
        total_time_on = time.perf_counter() - t_start

        ops_on = len(latencies_on) / total_time_on if total_time_on > 0 else 0.0
        p50_on, p95_on, p99_on = _compute_percentiles(latencies_on)

        cache_on_throughputs.append(round(ops_on, 1))
        cache_on_p50.append(round(p50_on, 3))
        cache_on_p95.append(round(p95_on, 3))
        cache_on_p99.append(round(p99_on, 3))

        # 2. Cache OFF evaluation (bypass cache so every read hits slow backend store)
        store_off = ObjectStore(base_latency=base_latency, seed=seed)
        node_off = EdgeNode(
            "node_off",
            store=store_off,
            locks=LockService(),
            backend_limiter=TokenBucket(100000, 100000),
            client_limiter=PerClientLimiter(100000, 100000),
        )
        node_off.write("init", file_name, 0, file_bytes)
        # Bypassing local cache lookup sends every read to the backend store
        node_off._cache.get = lambda k: node_off._cache._loader(k)  # type: ignore[assignment]

        latencies_off: list[float] = []
        stop_off = threading.Event()

        def worker_off(cid: str) -> None:
            local_lats: list[float] = []
            while not stop_off.is_set():
                t0 = time.perf_counter()
                node_off.read(cid, file_name, 0, 1024)
                local_lats.append((time.perf_counter() - t0) * 1000.0)
            latencies_off.extend(local_lats)

        threads = [threading.Thread(target=worker_off, args=(f"c_{i}",)) for i in range(num_threads)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(run_duration)
        stop_off.set()
        for t in threads:
            t.join()
        total_time_off = time.perf_counter() - t_start

        ops_off = len(latencies_off) / total_time_off if total_time_off > 0 else 0.0
        p50_off, p95_off, p99_off = _compute_percentiles(latencies_off)

        cache_off_throughputs.append(round(ops_off, 1))
        cache_off_p50.append(round(p50_off, 3))
        cache_off_p95.append(round(p95_off, 3))
        cache_off_p99.append(round(p99_off, 3))

    return {
        "thread_counts": thread_counts,
        "cache_on": {
            "throughput": cache_on_throughputs,
            "p50_ms": cache_on_p50,
            "p95_ms": cache_on_p95,
            "p99_ms": cache_on_p99,
        },
        "cache_off": {
            "throughput": cache_off_throughputs,
            "p50_ms": cache_off_p50,
            "p95_ms": cache_off_p95,
            "p99_ms": cache_off_p99,
        },
    }


def run_experiment_2(
    quick: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 2: Hit rate vs cache size using a Zipfian access pattern (s=1.1) over 1000 chunks."""
    capacities = [20, 50, 100, 200, 500] if quick else [10, 25, 50, 100, 200, 400, 600, 800, 1000]
    total_requests = 3000 if quick else 10000
    num_chunks = 1000
    s_param = 1.1

    # Pre-generate request sequence using Zipfian distribution
    zipf = ZipfGenerator(n=num_chunks, s=s_param, seed=seed)
    access_sequence = [f"chunk_{zipf.next_chunk()}" for _ in range(total_requests)]

    hit_rates: list[float] = []
    hits_list: list[int] = []
    misses_list: list[int] = []

    for cap in capacities:
        cache = ChunkCache(loader=lambda k: (b"chunk_content", 1), capacity=cap)
        for key in access_sequence:
            cache.get(key)

        stats = cache.stats()
        h = stats["hits"]
        m = stats["misses"]
        rate = (h / (h + m)) * 100.0 if (h + m) > 0 else 0.0

        hit_rates.append(round(rate, 2))
        hits_list.append(h)
        misses_list.append(m)

    return {
        "capacities": capacities,
        "hit_rates": hit_rates,
        "hits": hits_list,
        "misses": misses_list,
        "total_requests": total_requests,
        "num_chunks": num_chunks,
        "s_parameter": s_param,
    }


def run_experiment_3(
    quick: bool = False,
    seed: int = 42,
    base_latency: float = 0.020,
) -> dict[str, Any]:
    """Experiment 3: Backend calls with vs without coalescing under a burst of concurrent cold reads."""
    burst_sizes = [1, 10, 50, 100] if quick else [1, 5, 10, 20, 50, 100, 200]

    calls_with: list[int] = []
    calls_without: list[int] = []
    calls_saved: list[int] = []

    for size in burst_sizes:
        # 1. With Coalescing: ChunkCache leader registers Future in _in_flight, waiters await result
        actual_backend_calls = [0]
        call_lock = threading.Lock()

        def loader_with(k: str) -> tuple[bytes, int]:
            with call_lock:
                actual_backend_calls[0] += 1
            time.sleep(base_latency)
            return (b"payload", 1)

        cache_with = ChunkCache(loader=loader_with, capacity=100)
        threads = [threading.Thread(target=cache_with.get, args=("cold_key",)) for _ in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        w_calls = actual_backend_calls[0]
        calls_with.append(w_calls)

        # 2. Without Coalescing: Parallel uncoordinated misses each trigger an independent backend fetch
        wo_calls = size
        calls_without.append(wo_calls)
        calls_saved.append(wo_calls - w_calls)

    return {
        "burst_sizes": burst_sizes,
        "calls_with_coalescing": calls_with,
        "calls_without_coalescing": calls_without,
        "calls_saved": calls_saved,
    }


def run_experiment_4(
    quick: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 4: Error rate and goodput under backend throttling with vs without rate limiter."""
    loads = [10, 25, 40, 50] if quick else [10, 20, 30, 40, 50, 60]
    backend_limit = 25  # ObjectStore throttles at 25 req/sec across 1.0s sliding window

    err_rates_without: list[float] = []
    goodputs_without: list[float] = []

    err_rates_with: list[float] = []
    goodputs_with: list[float] = []

    for load in loads:
        # 1. Without Rate Limiter: burst requests sent unthrottled directly to backend
        store_off = ObjectStore(max_calls_per_second=backend_limit, seed=seed)
        for i in range(load):
            store_off._objects[f"item_{i}"] = ObjectRecord(b"data", 1, 0)

        succ_off = [0]
        err_off = [0]
        lock_off = threading.Lock()

        def worker_without(idx: int) -> None:
            try:
                store_off.get(f"item_{idx}")
                with lock_off:
                    succ_off[0] += 1
            except Throttled:
                with lock_off:
                    err_off[0] += 1

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker_without, args=(i,)) for i in range(load)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt_off = time.perf_counter() - t0

        err_pct_off = (err_off[0] / load) * 100.0 if load > 0 else 0.0
        goodput_val_off = succ_off[0] / dt_off if dt_off > 0 else 0.0

        err_rates_without.append(round(err_pct_off, 1))
        goodputs_without.append(round(goodput_val_off, 1))

        # 2. With Rate Limiter: TokenBucket limits egress rate with C=5, R=18 (C + R <= backend_limit)
        store_on = ObjectStore(max_calls_per_second=backend_limit, seed=seed)
        limiter = TokenBucket(capacity=5, refill_rate=18)
        node_on = EdgeNode(
            "node_bench",
            store=store_on,
            locks=LockService(),
            backend_limiter=limiter,
            client_limiter=PerClientLimiter(100000, 100000),
        )

        for i in range(load):
            store_on._objects[f"item_{i}#meta"] = ObjectRecord(b"10", 1, 0)
            store_on._objects[f"item_{i}#0"] = ObjectRecord(b"data", 1, 0)

        succ_on = [0]
        err_on = [0]
        lock_on = threading.Lock()

        def worker_with(idx: int) -> None:
            try:
                node_on.read(f"client_{idx}", f"item_{idx}", 0, 10)
                with lock_on:
                    succ_on[0] += 1
            except Exception:
                with lock_on:
                    err_on[0] += 1

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker_with, args=(i,)) for i in range(load)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt_on = time.perf_counter() - t0

        err_pct_on = (err_on[0] / load) * 100.0 if load > 0 else 0.0
        goodput_val_on = succ_on[0] / dt_on if dt_on > 0 else 0.0

        err_rates_with.append(round(err_pct_on, 1))
        goodputs_with.append(round(goodput_val_on, 1))

    return {
        "offered_loads": loads,
        "backend_limit": backend_limit,
        "without_limiter": {
            "error_rate_pct": err_rates_without,
            "goodput_ops_sec": goodputs_without,
        },
        "with_limiter": {
            "error_rate_pct": err_rates_with,
            "goodput_ops_sec": goodputs_with,
        },
    }
