"""In-memory object store simulating backend latency, failures, and concurrency control."""

from collections import deque
import random
import threading
from typing import Callable, NamedTuple

from miniedge.clock import Clock
from miniedge.errors import StaleFencingToken, Throttled, TransientBackendError, VersionConflict


class ObjectRecord(NamedTuple):
    """Stored object payload with version and fencing metadata."""

    data: bytes
    version: int
    max_fencing_token: int


class ObjectStore:
    """Thread-safe simulated object store with versioning, fencing, and fault injection."""

    def __init__(
        self,
        base_latency: float = 0.0,
        latency_jitter: float = 0.0,
        failure_rate: float = 0.0,
        seed: int | None = None,
        clock: Clock | Callable[[], float] | None = None,
        max_calls_per_second: int | None = None,
        throttle_rate: int | None = None,
    ) -> None:
        """Initialize the object store with configurable latency and error simulation."""
        self._base_latency = float(base_latency)
        self._latency_jitter = float(latency_jitter)
        self._failure_rate = float(failure_rate)
        self._outage = False
        self._max_calls_per_second = (
            max_calls_per_second if max_calls_per_second is not None else throttle_rate
        )

        self._rng = random.Random(seed)

        # Standardize clock injection: wrap raw callables to provide a consistent Clock API.
        if isinstance(clock, Clock):
            self._clock = clock
        elif callable(clock):
            self._clock = Clock(time_fn=clock)
        else:
            self._clock = Clock()

        self._lock = threading.Lock()
        self._objects: dict[str, ObjectRecord] = {}
        self._fencing_tokens: dict[str, int] = {}
        self._call_timestamps: deque[float] = deque()

        self._stats = {
            "gets": 0,
            "puts": 0,
            "failures": 0,
            "throttled": 0,
        }

    def set_outage(self, outage: bool) -> None:
        """Enable or disable full backend outage simulation."""
        with self._lock:
            self._outage = bool(outage)

    def set_failure_rate(self, failure_rate: float) -> None:
        """Update probability of transient backend failures (must be between 0.0 and 1.0)."""
        if not (0.0 <= failure_rate <= 1.0):
            raise ValueError("failure_rate must be between 0.0 and 1.0")
        with self._lock:
            self._failure_rate = float(failure_rate)

    def set_throttle_rate(self, max_calls_per_second: int | None) -> None:
        """Update or clear the sliding-window call rate limit."""
        with self._lock:
            self._max_calls_per_second = max_calls_per_second

    def stats(self) -> dict[str, int]:
        """Return a snapshot copy of operation, failure, and throttling counters."""
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset operation and error counters to zero."""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0

    def _prepare_call(self) -> float:
        """Validate arrival rate, outage, and transient faults, returning latency duration."""
        with self._lock:
            now = self._clock.now()

            # Evict timestamps older than 1 second to maintain sliding window.
            while self._call_timestamps and self._call_timestamps[0] <= now - 1.0:
                self._call_timestamps.popleft()

            if self._max_calls_per_second is not None:
                if len(self._call_timestamps) >= self._max_calls_per_second:
                    self._stats["throttled"] += 1
                    raise Throttled(
                        f"ObjectStore throttled: exceeded {self._max_calls_per_second} calls/sec"
                    )

            self._call_timestamps.append(now)

            if self._outage:
                self._stats["failures"] += 1
                raise TransientBackendError("ObjectStore outage in progress")

            if self._failure_rate > 0.0 and self._rng.random() < self._failure_rate:
                self._stats["failures"] += 1
                raise TransientBackendError("Simulated backend transient error")

            # Calculate sleep duration under lock because self._rng is not thread-safe.
            sleep_duration = self._base_latency
            if self._latency_jitter > 0.0:
                sleep_duration += self._rng.uniform(0.0, self._latency_jitter)

            return sleep_duration

    def get(self, key: str) -> tuple[bytes, int]:
        """Fetch the payload and version for key, simulating latency outside the lock."""
        sleep_duration = self._prepare_call()
        if sleep_duration > 0.0:
            # Sleep without holding internal lock to allow concurrent requests.
            self._clock.sleep(sleep_duration)

        with self._lock:
            record = self._objects.get(key)
            if record is None:
                self._stats["gets"] += 1
                raise KeyError(f"Key not found in object store: {key}")
            self._stats["gets"] += 1
            return record.data, record.version

    def get_record(self, key: str) -> ObjectRecord:
        """Return the complete ObjectRecord (data, version, max_fencing_token)."""
        with self._lock:
            record = self._objects.get(key)
            if record is None:
                raise KeyError(f"Key not found in object store: {key}")
            return record

    def put(
        self,
        key: str,
        data: bytes,
        expected_version: int | None = None,
        fencing_token: int | None = None,
    ) -> int:
        """Store payload with CAS version checking and fencing validation."""
        sleep_duration = self._prepare_call()
        if sleep_duration > 0.0:
            # Sleep without holding internal lock to allow concurrent requests.
            self._clock.sleep(sleep_duration)

        with self._lock:
            # Fencing check: reject writers presenting a token lower than highest recorded.
            if fencing_token is not None:
                highest_token = self._fencing_tokens.get(key, 0)
                if fencing_token < highest_token:
                    raise StaleFencingToken(
                        f"Stale fencing token {fencing_token} for key '{key}' "
                        f"(highest recorded is {highest_token})"
                    )
                self._fencing_tokens[key] = max(highest_token, fencing_token)

            record = self._objects.get(key)
            curr_version = record.version if record is not None else 0

            # CAS validation: mismatch against expected_version signals a conflicting write.
            if expected_version is not None and curr_version != expected_version:
                raise VersionConflict(
                    f"Version conflict for key '{key}': expected {expected_version}, "
                    f"found {curr_version}"
                )

            new_version = curr_version + 1
            max_token = self._fencing_tokens.get(key, 0)
            self._objects[key] = ObjectRecord(
                data=bytes(data),
                version=new_version,
                max_fencing_token=max_token,
            )
            self._stats["puts"] += 1
            return new_version

    def delete(self, key: str) -> bool:
        """Remove object for key, returning True if it existed."""
        sleep_duration = self._prepare_call()
        if sleep_duration > 0.0:
            self._clock.sleep(sleep_duration)

        with self._lock:
            existed = self._objects.pop(key, None) is not None
            return existed
