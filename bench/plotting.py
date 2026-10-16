"""Plotting module for benchmark results using matplotlib."""

import os
from typing import Any

import matplotlib

# Use headless backend to prevent GUI popups
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_experiment_1(data: dict[str, Any], output_path: str) -> None:
    """Plot latency and throughput vs client thread counts (Cache ON vs OFF)."""
    threads = data["thread_counts"]
    on_tput = data["cache_on"]["throughput"]
    off_tput = data["cache_off"]["throughput"]

    on_p50 = data["cache_on"]["p50_ms"]
    on_p95 = data["cache_on"]["p95_ms"]
    on_p99 = data["cache_on"]["p99_ms"]

    off_p50 = data["cache_off"]["p50_ms"]
    off_p95 = data["cache_off"]["p95_ms"]
    off_p99 = data["cache_off"]["p99_ms"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=150)

    # 1. Throughput subplot
    ax1.plot(threads, on_tput, "o-", color="#1f77b4", linewidth=2.0, markersize=6, label="Cache ON")
    ax1.plot(threads, off_tput, "s--", color="#d62728", linewidth=2.0, markersize=6, label="Cache OFF (Backend)")
    ax1.set_title("Experiment 1: Throughput vs Concurrent Client Threads", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Client Threads", fontsize=10)
    ax1.set_ylabel("Throughput (ops/sec)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Latency percentiles subplot (logarithmic scale due to 100x cache speedup)
    ax2.plot(threads, on_p50, "o-", color="#1f77b4", label="Cache ON p50")
    ax2.plot(threads, on_p95, "^--", color="#1f77b4", alpha=0.7, label="Cache ON p95")
    ax2.plot(threads, on_p99, "x:", color="#1f77b4", alpha=0.5, label="Cache ON p99")

    ax2.plot(threads, off_p50, "s-", color="#d62728", label="Cache OFF p50")
    ax2.plot(threads, off_p95, "v--", color="#d62728", alpha=0.7, label="Cache OFF p95")
    ax2.plot(threads, off_p99, "+:", color="#d62728", alpha=0.5, label="Cache OFF p99")

    ax2.set_title("Experiment 1: Latency Percentiles (p50/p95/p99) vs Threads", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Client Threads", fontsize=10)
    ax2.set_ylabel("Latency (ms, log scale)", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_2(data: dict[str, Any], output_path: str) -> None:
    """Plot cache hit rate vs cache capacity for Zipfian access pattern."""
    capacities = data["capacities"]
    hit_rates = data["hit_rates"]
    num_chunks = data["num_chunks"]
    s_param = data["s_parameter"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(capacities, hit_rates, "o-", color="#2ca02c", linewidth=2.5, markersize=7, label=f"Zipfian s={s_param}")

    ax.set_title(
        f"Experiment 2: Cache Hit Rate vs Capacity (Zipfian s={s_param}, N={num_chunks})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Cache Capacity (Chunks)", fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    # Annotate 20% capacity point
    for cap, hr in zip(capacities, hit_rates):
        if cap in (100, 200, 500):
            ax.annotate(
                f"{hr:.1f}%",
                (cap, hr),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_3(data: dict[str, Any], output_path: str) -> None:
    """Plot backend calls with vs without coalescing under concurrent cold read bursts."""
    burst_sizes = data["burst_sizes"]
    with_coal = data["calls_with_coalescing"]
    without_coal = data["calls_without_coalescing"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(burst_sizes, without_coal, "s--", color="#d62728", linewidth=2.0, markersize=6, label="Without Coalescing")
    ax.plot(burst_sizes, with_coal, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="With Coalescing")

    # Highlight saved backend calls
    ax.fill_between(burst_sizes, with_coal, without_coal, color="#1f77b4", alpha=0.15, label="Backend Calls Saved")

    ax.set_title("Experiment 3: Backend Calls vs Concurrent Cold Readers", fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrent Client Readers (Burst Size)", fontsize=10)
    ax.set_ylabel("Backend Calls Issued", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_4(data: dict[str, Any], output_path: str) -> None:
    """Plot error rate and goodput under backend throttling with vs without rate limiter."""
    loads = data["offered_loads"]
    backend_limit = data["backend_limit"]
    err_without = data["without_limiter"]["error_rate_pct"]
    err_with = data["with_limiter"]["error_rate_pct"]
    goodput_without = data["without_limiter"]["goodput_ops_sec"]
    goodput_with = data["with_limiter"]["goodput_ops_sec"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # 1. Error Rate subplot
    ax1.plot(loads, err_without, "s--", color="#d62728", linewidth=2.0, markersize=6, label="Without Limiter")
    ax1.plot(loads, err_with, "o-", color="#2ca02c", linewidth=2.0, markersize=6, label="With Limiter")
    ax1.axvline(backend_limit, color="black", linestyle=":", linewidth=1.5, label=f"Store Limit ({backend_limit}/s)")
    ax1.set_title("Error Rate (%) vs Offered Load", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Offered Load (req/sec)", fontsize=10)
    ax1.set_ylabel("Error Rate (%)", fontsize=10)
    ax1.set_ylim(-5, 100)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Goodput subplot
    ax2.plot(loads, goodput_without, "s--", color="#d62728", linewidth=2.0, markersize=6, label="Without Limiter")
    ax2.plot(loads, goodput_with, "o-", color="#2ca02c", linewidth=2.0, markersize=6, label="With Limiter")
    ax2.set_title("Goodput (Successful ops/sec) vs Offered Load", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Offered Load (req/sec)", fontsize=10)
    ax2.set_ylabel("Goodput (ops/sec)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_all(results: dict[str, Any], output_dir: str) -> list[str]:
    """Render all 4 benchmark graphs into output_dir and return created filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    generated: list[str] = []

    p1 = os.path.join(output_dir, "exp1_latency_throughput.png")
    plot_experiment_1(results["experiment_1"], p1)
    generated.append(p1)

    p2 = os.path.join(output_dir, "exp2_cache_hit_rate.png")
    plot_experiment_2(results["experiment_2"], p2)
    generated.append(p2)

    p3 = os.path.join(output_dir, "exp3_request_coalescing.png")
    plot_experiment_3(results["experiment_3"], p3)
    generated.append(p3)

    p4 = os.path.join(output_dir, "exp4_throttling_limiter.png")
    plot_experiment_4(results["experiment_4"], p4)
    generated.append(p4)

    return generated
