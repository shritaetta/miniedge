"""Benchmark runner: executes MiniEdge performance experiments, saves summary.json, and produces plots."""

import argparse
import json
import os
import sys
import time
from typing import Any

from bench.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
)
from bench.plotting import plot_all


def print_markdown_report(results: dict[str, Any]) -> None:
    """Print formatted markdown tables for each benchmark experiment to stdout."""
    exp1 = results["experiment_1"]
    exp2 = results["experiment_2"]
    exp3 = results["experiment_3"]
    exp4 = results["experiment_4"]

    print("\n# MiniEdge Benchmark Evaluation Report\n")

    # Table 1: Latency & Throughput
    print("## 1. Latency and Throughput vs Client Threads (Cache ON vs OFF)")
    print("| Threads | Cache ON (ops/sec) | ON p50 (ms) | ON p95 (ms) | ON p99 (ms) | Cache OFF (ops/sec) | OFF p50 (ms) | OFF p95 (ms) | OFF p99 (ms) |")
    print("|---------|--------------------|-------------|-------------|-------------|---------------------|--------------|--------------|--------------|")
    for i, t in enumerate(exp1["thread_counts"]):
        on_tput = exp1["cache_on"]["throughput"][i]
        on_p50 = exp1["cache_on"]["p50_ms"][i]
        on_p95 = exp1["cache_on"]["p95_ms"][i]
        on_p99 = exp1["cache_on"]["p99_ms"][i]

        off_tput = exp1["cache_off"]["throughput"][i]
        off_p50 = exp1["cache_off"]["p50_ms"][i]
        off_p95 = exp1["cache_off"]["p95_ms"][i]
        off_p99 = exp1["cache_off"]["p99_ms"][i]

        print(
            f"| {t:7d} | {on_tput:18.1f} | {on_p50:11.3f} | {on_p95:11.3f} | {on_p99:11.3f} | "
            f"{off_tput:19.1f} | {off_p50:12.3f} | {off_p95:12.3f} | {off_p99:12.3f} |"
        )
    print()

    # Table 2: Cache Hit Rate vs Cache Size
    print("## 2. Hit Rate vs Cache Capacity (Zipfian s=1.1 over 1000 Chunks)")
    print("| Cache Capacity (Chunks) | Requests | Hits | Misses | Hit Rate (%) |")
    print("|-------------------------|----------|------|--------|--------------|")
    for i, cap in enumerate(exp2["capacities"]):
        hr = exp2["hit_rates"][i]
        h = exp2["hits"][i]
        m = exp2["misses"][i]
        print(f"| {cap:23d} | {exp2['total_requests']:8d} | {h:4d} | {m:6d} | {hr:11.2f}% |")
    print()

    # Table 3: Request Coalescing
    print("## 3. Backend Calls With vs Without Request Coalescing")
    print("| Concurrent Readers (Burst) | Calls With Coalescing | Calls Without Coalescing | Backend Calls Saved |")
    print("|----------------------------|-----------------------|--------------------------|---------------------|")
    for i, burst in enumerate(exp3["burst_sizes"]):
        w = exp3["calls_with_coalescing"][i]
        wo = exp3["calls_without_coalescing"][i]
        saved = exp3["calls_saved"][i]
        print(f"| {burst:26d} | {w:21d} | {wo:24d} | {saved:19d} |")
    print()

    # Table 4: Backend Throttling
    print("## 4. Error Rate and Goodput under Backend Throttling (Limit = 25 req/sec)")
    print("| Offered Load (req/sec) | Without Limiter Error (%) | Without Limiter Goodput | With Limiter Error (%) | With Limiter Goodput |")
    print("|------------------------|---------------------------|-------------------------|------------------------|----------------------|")
    for i, load in enumerate(exp4["offered_loads"]):
        err_wo = exp4["without_limiter"]["error_rate_pct"][i]
        gp_wo = exp4["without_limiter"]["goodput_ops_sec"][i]
        err_w = exp4["with_limiter"]["error_rate_pct"][i]
        gp_w = exp4["with_limiter"]["goodput_ops_sec"][i]
        print(f"| {load:22d} | {err_wo:24.1f}% | {gp_wo:23.1f} | {err_w:21.1f}% | {gp_w:20.1f} |")
    print()


def main() -> None:
    """Parse CLI arguments, run benchmark suite, write summary.json, and render plots."""
    parser = argparse.ArgumentParser(description="MiniEdge Performance Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run shortened benchmarks for fast evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs (default: 42)")
    parser.add_argument("--latency", type=float, default=0.020, help="Base backend simulated latency in seconds (default: 0.020)")
    parser.add_argument("--output-dir", type=str, default="bench/results", help="Directory to save summary and plots")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting MiniEdge Benchmark Suite (quick={args.quick}, seed={args.seed}, latency={args.latency}s)...")
    t0 = time.perf_counter()

    # Execute all 4 experiments
    print("-> Running Experiment 1: Latency & Throughput vs Threads...")
    exp1 = run_experiment_1(quick=args.quick, seed=args.seed, base_latency=args.latency)

    print("-> Running Experiment 2: Cache Hit Rate vs Capacity (Zipfian)...")
    exp2 = run_experiment_2(quick=args.quick, seed=args.seed)

    print("-> Running Experiment 3: Request Coalescing under Cold Bursts...")
    exp3 = run_experiment_3(quick=args.quick, seed=args.seed, base_latency=args.latency)

    print("-> Running Experiment 4: Error Rate & Goodput under Backend Throttling...")
    exp4 = run_experiment_4(quick=args.quick, seed=args.seed)

    total_elapsed = time.perf_counter() - t0
    print(f"All experiments completed in {total_elapsed:.2f} seconds.")

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quick": args.quick,
            "seed": args.seed,
            "simulated_latency_sec": args.latency,
            "total_elapsed_sec": round(total_elapsed, 2),
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
    }

    # Write summary.json
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Raw numbers saved to: {summary_path}")

    # Generate matplotlib plots
    print("-> Rendering plots with matplotlib...")
    created_plots = plot_all(results, output_dir)
    for p in created_plots:
        print(f"   Created plot: {p}")

    # Output markdown report to stdout
    print_markdown_report(results)


if __name__ == "__main__":
    main()
