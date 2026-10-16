"""Unit tests for Clock and FakeClock."""

import threading
import time
import pytest

from miniedge.clock import Clock, FakeClock


def test_clock_default_behavior() -> None:
    """Verify standard Clock delegates to time.monotonic."""
    clock = Clock()
    t1 = clock.now()
    assert clock.monotonic() >= t1
    assert clock() >= t1
    clock.sleep(0)


def test_fake_clock_advance() -> None:
    """Verify FakeClock.advance moves time forward deterministically."""
    clock = FakeClock(initial_time=100.0)
    assert clock.now() == 100.0
    assert clock.monotonic() == 100.0
    assert clock() == 100.0

    clock.advance(10.5)
    assert clock.now() == 110.5

    clock.advance(0.0)
    assert clock.now() == 110.5

    with pytest.raises(ValueError, match="cannot be moved backwards"):
        clock.advance(-1.0)


def test_fake_clock_sleep_and_wake() -> None:
    """Verify FakeClock.advance wakes a thread sleeping until a future time."""
    clock = FakeClock(initial_time=0.0)
    wake_event = threading.Event()

    def sleeper() -> None:
        clock.sleep(5.0)
        wake_event.set()

    thread = threading.Thread(target=sleeper)
    thread.start()

    # Ensure thread has started and entered sleep before advancing clock.
    assert clock.wait_for_sleepers(count=1, timeout=2.0)
    assert not wake_event.is_set()

    # Advance partially; the sleeper should still be blocked.
    clock.advance(3.0)
    time.sleep(0.05)
    assert not wake_event.is_set()
    assert thread.is_alive()

    # Advance past deadline; sleeper must wake up and terminate cleanly.
    clock.advance(2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert wake_event.is_set()
    assert clock.sleeper_count == 0


def test_fake_clock_multiple_sleepers() -> None:
    """Verify FakeClock wakes sleepers in the correct order as time progresses."""
    clock = FakeClock(initial_time=0.0)
    first_woken = threading.Event()
    second_woken = threading.Event()

    def sleep_first() -> None:
        clock.sleep(10.0)
        first_woken.set()

    def sleep_second() -> None:
        clock.sleep(25.0)
        second_woken.set()

    t1 = threading.Thread(target=sleep_first)
    t2 = threading.Thread(target=sleep_second)
    t1.start()
    t2.start()

    assert clock.wait_for_sleepers(count=2, timeout=2.0)

    # Advance to 10.0: first thread wakes, second stays asleep.
    clock.advance(10.0)
    t1.join(timeout=2.0)
    assert first_woken.is_set()
    assert not second_woken.is_set()
    assert t2.is_alive()

    # Advance another 15.0: second thread wakes.
    clock.advance(15.0)
    t2.join(timeout=2.0)
    assert second_woken.is_set()
    assert not t2.is_alive()
