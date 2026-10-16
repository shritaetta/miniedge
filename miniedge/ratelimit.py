"""Token bucket rate limiters and retry policy with full jitter backoff."""

import random
import threading
from typing import Any, Callable, TypeVar

from miniedge.clock import Clock, FakeClock
from miniedge.errors import Throttled, TransientBackendError

T = TypeVar("T")


class TokenBucket:
    """Thread-safe token bucket rate limiter with lazy refills and starvation-free queuing.

    Invariants maintained under self._lock:
    1. self._tokens represents immediately available free tokens (0.0 <= _tokens <= capacity).
    2. self._next_available is the virtual reservation timeline: the earliest timestamp when
       all previously queued acquire() reservations are fulfilled.
    3. Whenever self._next_available > now, all accumulated tokens up to that future timestamp
       have been pre-allocated to sleeping waiters in strict FIFO order, so self._tokens MUST equal 0.0.
    4. Once now >= self._next_available, prior reservations have completed, and tokens refill
       continuously from self._next_available up to now, capped at self._capacity.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        clock: Clock | None = None,
    ) -> None:
        """Initialize bucket with burst capacity, tokens-per-second refill rate, and clock."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)

        # Standardize clock injection for deterministic timing tests.
        if isinstance(clock, Clock):
            self._clock = clock
        elif callable(clock):
            self._clock = Clock(time_fn=clock)
        else:
            self._clock = Clock()

        self._lock = threading.Lock()
        self._tokens: float = float(capacity)
        # Tracks the earliest timestamp when all previously queued reservations are satisfied.
        self._next_available: float = self._clock.now()

    def _assert_invariants(self, now: float) -> None:
        """Debug assertion verifying tokens == 0.0 whenever reservations exist in the future."""
        assert not (self._next_available > now and self._tokens > 0.0), (
            f"Invariant violation: tokens={self._tokens} must be 0 when "
            f"next_available ({self._next_available}) > now ({now})"
        )

    @property
    def tokens(self) -> float:
        """Return the current number of available tokens after lazy refill."""
        with self._lock:
            now = self._clock.now()
            if now > self._next_available:
                refilled = (now - self._next_available) * self._refill_rate
                self._tokens = min(self._capacity, self._tokens + refilled)
                self._next_available = now
            self._assert_invariants(now)
            return self._tokens

    def try_acquire(self, n: float = 1) -> bool:
        """Attempt to acquire n tokens non-blocking, returning True if successful."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n > self._capacity:
            raise ValueError(f"Requested {n} tokens exceeds bucket capacity {self._capacity}")

        with self._lock:
            now = self._clock.now()
            # Catch up on tokens generated since the last reservation cleared.
            if now > self._next_available:
                refilled = (now - self._next_available) * self._refill_rate
                self._tokens = min(self._capacity, self._tokens + refilled)
                self._next_available = now

            # Do not allow non-blocking acquisition to steal tokens promised to sleeping threads.
            if now < self._next_available or self._tokens < n:
                self._assert_invariants(now)
                return False

            self._tokens -= n
            self._assert_invariants(now)
            return True

    def acquire(self, n: float = 1, timeout: float | None = None) -> bool:
        """Acquire n tokens, sleeping outside the lock until available or timeout expires.

        If the deficit cannot be satisfied within `timeout`, returns False immediately
        without reserving a slot. If satisfied within `timeout`, reserves the slot in the
        queue and sleeps outside the lock for the calculated wait time.

        Note on abort-during-sleep behavior: If a thread is aborted, cancelled, or
        interrupted while sleeping after its slot has been reserved, the reservation is
        NOT rolled back. The virtual timeline (_next_available) remains advanced, meaning
        subsequent callers will still wait past that slot as if it were consumed.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n > self._capacity:
            raise ValueError(f"Requested {n} tokens exceeds bucket capacity {self._capacity}")

        with self._lock:
            now = self._clock.now()
            # Catch up on tokens generated since the last reservation cleared.
            if now > self._next_available:
                refilled = (now - self._next_available) * self._refill_rate
                self._tokens = min(self._capacity, self._tokens + refilled)
                self._next_available = now

            if self._tokens >= n and self._next_available == now:
                # Immediate acquisition without sleep when bucket has sufficient tokens.
                self._tokens -= n
                self._assert_invariants(now)
                return True

            # Compute deficit from next_available to ensure strict FIFO reservation fairness.
            if self._next_available == now:
                deficit = n - self._tokens
                self._tokens = 0.0
            else:
                deficit = n

            wait_time = (self._next_available - now) + (deficit / self._refill_rate)

            if timeout is not None and wait_time > timeout:
                # Restore unconsumed tokens if reservation could not be fulfilled within timeout.
                if self._next_available == now:
                    self._tokens += (n - deficit)
                self._assert_invariants(now)
                return False

            # Reserve this thread's tokens so later callers cannot jump ahead in queue.
            self._next_available += (deficit / self._refill_rate)
            self._assert_invariants(now)

        # Sleep outside the lock so backend throughput and other threads are never blocked.
        if wait_time > 0:
            self._clock.sleep(wait_time)

        return True


class PerClientLimiter:
    """Thread-safe rate limiter managing lazy per-client token buckets."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        clock: Clock | None = None,
    ) -> None:
        """Initialize limiter with default capacity and refill rate per client."""
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[Any, TokenBucket] = {}

    def allow(self, client_id: Any) -> bool:
        """Check if client has an available token, creating its bucket lazily."""
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self._capacity,
                    refill_rate=self._refill_rate,
                    clock=self._clock,
                )
                self._buckets[client_id] = bucket

        return bucket.try_acquire(1)


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base: float = 0.05,
    cap: float = 2.0,
    retry_on: tuple[type[Exception], ...] | type[Exception] = (Throttled, TransientBackendError),
    clock: Clock | None = None,
    rng: random.Random | None = None,
) -> T:
    """Execute fn with exponential backoff and full jitter retry on transient exceptions."""
    if clock is None:
        clock = Clock()
    if rng is None:
        rng = random.Random()

    retry_tuple = (retry_on,) if isinstance(retry_on, type) else tuple(retry_on)

    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_tuple:
            if attempt >= retries:
                raise

            # Full jitter formula spreads out retrying clients to prevent synchronized thundering herds.
            max_backoff = min(cap, base * (2**attempt))
            sleep_duration = rng.uniform(0.0, max_backoff)

            # Avoid deadlocking synchronous single-threaded callers when FakeClock is injected.
            if isinstance(clock, FakeClock):
                clock.advance(sleep_duration)
            else:
                clock.sleep(sleep_duration)
