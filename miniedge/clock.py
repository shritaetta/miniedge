"""Clock abstractions providing injectable monotonic time and sleep."""

import threading
import time
from typing import Callable


class Clock:
    """Wraps monotonic time and sleep for deterministic dependency injection."""

    def __init__(
        self,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize with injectable time and sleep callables."""
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    def now(self) -> float:
        """Return the current monotonic time in seconds."""
        return self._time_fn()

    def monotonic(self) -> float:
        """Return the current monotonic time in seconds (standard library alias)."""
        return self._time_fn()

    def sleep(self, seconds: float) -> None:
        """Pause execution for the specified duration in seconds."""
        if seconds > 0:
            self._sleep_fn(seconds)

    def register_condition(self, cond: threading.Condition) -> None:
        """Register a condition variable to be notified on time advances (no-op on wall clock)."""
        pass

    def unregister_condition(self, cond: threading.Condition) -> None:
        """Unregister a condition variable (no-op on wall clock)."""
        pass

    def __call__(self) -> float:
        """Return current time when called directly, matching time.monotonic signature."""
        return self._time_fn()


class FakeClock(Clock):
    """Controllable clock for deterministic testing of time-dependent logic."""

    def __init__(self, initial_time: float = 0.0) -> None:
        """Initialize fake clock with starting timestamp."""
        self._now = float(initial_time)
        self._cond = threading.Condition()
        self._sleepers = 0
        self._registered_conditions: set[threading.Condition] = set()
        super().__init__(time_fn=self.now, sleep_fn=self.sleep)

    def register_condition(self, cond: threading.Condition) -> None:
        """Register a condition to be notified whenever simulated time advances."""
        with self._cond:
            self._registered_conditions.add(cond)

    def unregister_condition(self, cond: threading.Condition) -> None:
        """Unregister a condition so it is no longer notified on simulated time advances."""
        with self._cond:
            self._registered_conditions.discard(cond)

    def now(self) -> float:
        """Return the current simulated timestamp in seconds."""
        with self._cond:
            return self._now

    def advance(self, seconds: float) -> None:
        """Move simulated time forward and wake any sleepers whose deadline has arrived."""
        if seconds < 0:
            raise ValueError("Clock cannot be moved backwards")
        with self._cond:
            self._now += seconds
            # Wake all waiting sleepers so those whose deadline expired can proceed.
            self._cond.notify_all()
            conds = list(self._registered_conditions)

        # Notify registered conditions without holding self._cond to avoid lock-order inversion.
        for cond in conds:
            with cond:
                cond.notify_all()

    def sleep(self, seconds: float) -> None:
        """Block until the simulated clock advances past the target deadline."""
        if seconds <= 0:
            return
        with self._cond:
            target = self._now + seconds
            self._sleepers += 1
            # Signal wait_for_sleepers so tests can coordinate without polling.
            self._cond.notify_all()
            try:
                while self._now < target:
                    self._cond.wait()
            finally:
                self._sleepers -= 1

    @property
    def sleeper_count(self) -> int:
        """Return the number of threads currently blocked in sleep()."""
        with self._cond:
            return self._sleepers

    def wait_for_sleepers(self, count: int, timeout: float = 5.0) -> bool:
        """Block until at least count threads are waiting in sleep()."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._sleepers < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True
