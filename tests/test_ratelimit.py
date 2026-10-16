"""Unit and concurrency tests for TokenBucket, PerClientLimiter, and retry_with_backoff."""

import random
import threading
import pytest

from miniedge.clock import Clock, FakeClock
from miniedge.errors import Throttled, TransientBackendError
from miniedge.ratelimit import PerClientLimiter, TokenBucket, retry_with_backoff


def test_token_bucket_exact_refill_and_burst() -> None:
    """Verify burst capacity consumption, exact fractional refill math, and capacity cap."""
    clock = FakeClock(initial_time=0.0)
    bucket = TokenBucket(capacity=5.0, refill_rate=2.0, clock=clock)

    assert bucket.tokens == 5.0

    # Burst down to empty.
    for _ in range(5):
        assert bucket.try_acquire(1.0) is True

    # Empty bucket rejects next non-blocking request.
    assert bucket.try_acquire(1.0) is False
    assert bucket.tokens == 0.0

    # Advance 0.5s: exactly 0.5 * 2.0 = 1.0 token refilled.
    clock.advance(0.5)
    assert bucket.tokens == 1.0
    assert bucket.try_acquire(1.0) is True
    assert bucket.try_acquire(1.0) is False

    # Advance 10.0s: theoretical refill is 20, but capacity is capped at 5.0.
    clock.advance(10.0)
    assert bucket.tokens == 5.0

    # Test invalid acquisition sizes.
    with pytest.raises(ValueError, match="must be positive"):
        bucket.try_acquire(0)

    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        bucket.try_acquire(6.0)


def test_token_bucket_acquire_timeout() -> None:
    """Verify acquire() with timeout returns False when deficit cannot be satisfied in time."""
    clock = FakeClock(initial_time=0.0)
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0, clock=clock)

    # Empty the bucket.
    assert bucket.try_acquire(10.0) is True

    # 5 tokens at 2 tokens/sec requires 2.5s. Timeout of 1.0s cannot satisfy it.
    assert bucket.acquire(n=5.0, timeout=1.0) is False

    # Ensure unconsumed deficit was not left committed.
    clock.advance(2.5)
    assert bucket.tokens == 5.0


def test_token_bucket_20_threads_within_capacity_and_rate() -> None:
    """Verify 20 concurrent threads never exceed capacity + rate * elapsed tokens."""
    clock = FakeClock(initial_time=0.0)
    capacity = 10.0
    refill_rate = 20.0
    bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate, clock=clock)

    num_threads = 20
    tokens_granted = 0
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker() -> None:
        nonlocal tokens_granted
        while not stop_event.is_set():
            if bucket.acquire(1.0, timeout=0.5):
                with lock:
                    tokens_granted += 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    # Step through simulated time across 2.0 seconds.
    total_steps = 20
    step_duration = 0.1
    for _ in range(total_steps):
        # Allow threads to reach their sleep points on FakeClock.
        clock.wait_for_sleepers(count=1, timeout=0.5)
        clock.advance(step_duration)

    stop_event.set()
    # Unblock any remaining timed-out sleepers so threads exit cleanly.
    clock.advance(1.0)

    for thread in threads:
        thread.join(timeout=5.0)

    elapsed_time = total_steps * step_duration
    max_allowed_tokens = capacity + (refill_rate * elapsed_time)
    assert tokens_granted <= max_allowed_tokens, (
        f"Granted {tokens_granted} tokens, which exceeds theoretical max {max_allowed_tokens}"
    )


def test_per_client_limiter_lazy_and_isolated() -> None:
    """Verify PerClientLimiter creates independent buckets per client lazily."""
    clock = FakeClock(initial_time=0.0)
    limiter = PerClientLimiter(capacity=2.0, refill_rate=1.0, clock=clock)

    # Client A consumes its quota.
    assert limiter.allow("client_A") is True
    assert limiter.allow("client_A") is True
    assert limiter.allow("client_A") is False

    # Client B has independent capacity.
    assert limiter.allow("client_B") is True
    assert limiter.allow("client_B") is True
    assert limiter.allow("client_B") is False

    # Advance clock by 1.0s: both clients refill 1 token.
    clock.advance(1.0)
    assert limiter.allow("client_A") is True
    assert limiter.allow("client_A") is False
    assert limiter.allow("client_B") is True
    assert limiter.allow("client_B") is False


def test_retry_with_backoff_success_after_two_failures() -> None:
    """Verify retry_with_backoff retries transient failures and returns on success."""
    clock = FakeClock()
    calls = 0

    def flaky_fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientBackendError(f"Temporary outage {calls}")
        return "success_data"

    result = retry_with_backoff(flaky_fn, retries=5, clock=clock)
    assert result == "success_data"
    assert calls == 3


def test_retry_with_backoff_gives_up_after_n_retries() -> None:
    """Verify retry_with_backoff gives up after N retries and re-raises the final error."""
    clock = FakeClock()
    calls = 0

    def always_throttled() -> None:
        nonlocal calls
        calls += 1
        raise Throttled("429 rate limit")

    with pytest.raises(Throttled, match="429 rate limit"):
        retry_with_backoff(always_throttled, retries=3, clock=clock)

    # Initial attempt + 3 retries = 4 total calls.
    assert calls == 4


def test_retry_with_backoff_does_not_retry_unrelated_exceptions() -> None:
    """Verify retry_with_backoff immediately propagates unconfigured exceptions without retrying."""
    clock = FakeClock()
    calls = 0

    def bad_request() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("Invalid argument")

    with pytest.raises(ValueError, match="Invalid argument"):
        retry_with_backoff(bad_request, retries=5, clock=clock)

    assert calls == 1


def test_retry_with_backoff_sleep_bounds_with_seeded_rng() -> None:
    """Verify full jitter sleep durations remain strictly within exponential backoff bounds."""
    sleeps: list[float] = []
    clock = Clock(sleep_fn=sleeps.append)
    rng = random.Random(42)

    def always_fails() -> None:
        raise TransientBackendError("fail")

    base = 0.05
    cap = 1.0
    retries = 4

    with pytest.raises(TransientBackendError):
        retry_with_backoff(
            always_fails,
            retries=retries,
            base=base,
            cap=cap,
            clock=clock,
            rng=rng,
        )

    assert len(sleeps) == retries
    for attempt, sleep_dur in enumerate(sleeps):
        upper_bound = min(cap, base * (2**attempt))
        assert 0.0 <= sleep_dur <= upper_bound, (
            f"Attempt {attempt} slept {sleep_dur}, exceeding bound {upper_bound}"
        )


def test_try_acquire_fails_while_acquire_waiter_holds_reservation() -> None:
    """Verify try_acquire() returns False and cannot barge in while an acquire() waiter holds a reservation."""
    clock = FakeClock(initial_time=0.0)
    bucket = TokenBucket(capacity=5.0, refill_rate=1.0, clock=clock)

    # Drain the bucket completely.
    assert bucket.try_acquire(5.0) is True

    # Waiter requests 3 tokens (requires 3.0 seconds to refill).
    waiter_result: list[bool] = []
    t = threading.Thread(target=lambda: waiter_result.append(bucket.acquire(3.0)))
    t.start()

    # Wait until waiter has calculated deficit and entered sleep on FakeClock.
    assert clock.wait_for_sleepers(count=1, timeout=2.0)

    # Non-blocking acquisition must return False: cannot steal tokens pledged to sleeping waiter.
    assert bucket.try_acquire(1.0) is False

    # Advance time partially (1.5s): 1.5 tokens refilled, but still reserved for the waiter.
    clock.advance(1.5)
    assert bucket.try_acquire(1.0) is False

    # Advance past reservation (remaining 1.5s): waiter wakes up and completes.
    clock.advance(1.5)
    t.join(timeout=2.0)
    assert waiter_result == [True]

    # Subsequent advance refills unreserved tokens for immediate acquisition.
    clock.advance(1.0)
    assert bucket.try_acquire(1.0) is True


def test_failed_acquire_timeout_leaves_later_callers_wait_times_unchanged() -> None:
    """Verify that an acquire(timeout) call that gives up does not advance queue or delay later callers."""
    clock = FakeClock(initial_time=0.0)
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0, clock=clock)

    # Drain bucket completely.
    assert bucket.try_acquire(10.0) is True

    # Caller 1 asks for 6 tokens with a timeout of 1.0s.
    # Deficit is 6 tokens, which at 2 tokens/sec takes 3.0s. Since 3.0s > 1.0s, it must fail immediately.
    assert bucket.acquire(6.0, timeout=1.0) is False

    # Now, verify that Caller 1 did NOT reserve a slot or delay Caller 2.
    # Caller 2 asks for 2 tokens with timeout=1.0s.
    # If Caller 1 had delayed the queue by 3s, Caller 2 would see wait_time=4s > 1.0s and fail.
    # Because Caller 1 did not delay the queue, Caller 2 needs exactly 2 tokens / 2.0 = 1.0s <= timeout.
    caller2_result: list[bool] = []
    t = threading.Thread(target=lambda: caller2_result.append(bucket.acquire(2.0, timeout=1.0)))
    t.start()

    assert clock.wait_for_sleepers(count=1, timeout=2.0)

    # Advance clock by exactly 1.0s: Caller 2 must wake up and succeed.
    clock.advance(1.0)
    t.join(timeout=2.0)
    assert caller2_result == [True]


def test_fuzz_concurrent_token_bucket_against_fake_clock() -> None:
    """Fuzz 20 threads mixing try_acquire, acquire, and timed acquire against FakeClock."""
    clock = FakeClock(initial_time=0.0)
    capacity = 20.0
    refill_rate = 50.0
    bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate, clock=clock)

    num_threads = 20
    total_granted = 0.0
    lock = threading.Lock()
    stop_event = threading.Event()
    errors: list[Exception] = []

    def worker(seed: int) -> None:
        nonlocal total_granted
        rnd = random.Random(seed)
        try:
            while not stop_event.is_set():
                op = rnd.randint(0, 2)
                n = float(rnd.randint(1, 3))
                if op == 0:
                    if bucket.try_acquire(n):
                        with lock:
                            total_granted += n
                elif op == 1:
                    timeout = rnd.uniform(0.02, 0.1)
                    if bucket.acquire(n, timeout=timeout):
                        with lock:
                            total_granted += n
                else:
                    if bucket.acquire(n):
                        with lock:
                            total_granted += n
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(1000 + i,))
        for i in range(num_threads)
    ]
    for thread in threads:
        thread.start()

    total_steps = 30
    step_duration = 0.1
    for _ in range(total_steps):
        clock.wait_for_sleepers(count=1, timeout=0.1)
        clock.advance(step_duration)

    stop_event.set()
    # Advance time to allow remaining sleepers to wake up and exit.
    clock.advance(5.0)

    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors, f"Unexpected exceptions during fuzz test: {errors}"

    elapsed_time = total_steps * step_duration
    max_possible_tokens = capacity + (refill_rate * elapsed_time)

    assert total_granted <= max_possible_tokens, (
        f"Total granted {total_granted} exceeded bound {max_possible_tokens}"
    )
    # Verify tokens never go negative via public property.
    assert bucket.tokens >= 0.0
