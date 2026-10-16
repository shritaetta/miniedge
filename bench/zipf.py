"""Zipfian distribution generator for cache workload modeling."""

import bisect
import random


class ZipfGenerator:
    """Discrete Zipfian distribution generator over N items with skew parameter s."""

    def __init__(self, n: int, s: float = 1.1, seed: int = 42) -> None:
        """Precompute cumulative distribution function (CDF) for discrete Zipfian selection."""
        if n < 1:
            raise ValueError("n must be at least 1")
        if s <= 0:
            raise ValueError("s must be positive")

        self.n = n
        self.s = s
        self._rng = random.Random(seed)

        # Precompute normalized CDF: P(k) proportional to 1 / (k^s)
        weights = [1.0 / (i**s) for i in range(1, n + 1)]
        total_weight = sum(weights)

        self._cdf: list[float] = []
        cumulative = 0.0
        for w in weights:
            cumulative += w / total_weight
            self._cdf.append(cumulative)
        self._cdf[-1] = 1.0

    def next_chunk(self) -> int:
        """Sample next item index in range [0, n - 1] using binary search over the CDF."""
        uniform_random = self._rng.random()
        return bisect.bisect_left(self._cdf, uniform_random)
