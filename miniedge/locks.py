"""Distributed lease-based lock service with fencing tokens and writer preference."""

from dataclasses import dataclass
import threading
from typing import Literal

from miniedge.clock import Clock
from miniedge.errors import LeaseExpired, LockTimeout

LockMode = Literal["shared", "exclusive"]


@dataclass(frozen=True)
class Lease:
    """Represents a time-bounded lease grant on a resource path."""

    path: str
    mode: LockMode
    holder_id: str
    token: int
    expires_at: float


class _PathLock:
    """Internal synchronization state and wait queue for a single resource path."""

    def __init__(self, path: str, initial_token: int = 0) -> None:
        self.path = path
        # Dedicated lock and Condition per path to isolate contention between different paths.
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.fencing_token = initial_token
        self.exclusive_holder: str | None = None
        self.exclusive_token: int = 0
        self.exclusive_expires_at: float = 0.0
        # Maps holder_id -> (expires_at, token)
        self.shared_holders: dict[str, tuple[float, int]] = {}
        self.waiting_writers = 0
        self.last_active_at: float = 0.0

    def purge_expired(self, now: float) -> bool:
        """Purge expired exclusive and shared leases lazily, returning True if any were removed."""
        changed = False
        if self.exclusive_holder is not None and self.exclusive_expires_at <= now:
            self.exclusive_holder = None
            self.exclusive_token = 0
            self.exclusive_expires_at = 0.0
            changed = True

        expired_shared = [h for h, (exp, _) in self.shared_holders.items() if exp <= now]
        for h in expired_shared:
            del self.shared_holders[h]
            changed = True

        if changed:
            # Wake all waiters so they can re-check availability following an expiry.
            self.cond.notify_all()
        return changed

    def calc_wait_timeout(self, now: float, deadline: float | None) -> float | None:
        """Calculate next wake timeout based on earliest lease expiry and caller deadline."""
        earliest_expiry: float | None = None
        if self.exclusive_holder is not None and self.exclusive_expires_at > now:
            earliest_expiry = self.exclusive_expires_at

        if self.shared_holders:
            future_shared = [exp for exp, _ in self.shared_holders.values() if exp > now]
            if future_shared:
                min_s = min(future_shared)
                if earliest_expiry is None or min_s < earliest_expiry:
                    earliest_expiry = min_s

        timeout: float | None = None
        if earliest_expiry is not None:
            timeout = max(0.001, earliest_expiry - now)

        if deadline is not None:
            rem = max(0.0, deadline - now)
            if timeout is not None:
                return min(rem, timeout)
            return rem
        return timeout

    def is_idle(self) -> bool:
        """Return True if path lock has no active holders and no waiting writers or condition waiters."""
        return (
            self.exclusive_holder is None
            and len(self.shared_holders) == 0
            and self.waiting_writers == 0
            and len(getattr(self.cond, "_waiters", [])) == 0
        )


class LockService:
    """In-process leased lock authority with fencing tokens, lazy expiry, and writer preference."""

    def __init__(self, default_ttl: float = 5.0, clock: Clock | None = None) -> None:
        """Initialize lock service with default lease TTL and injectable clock."""
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self._default_ttl = float(default_ttl)

        if isinstance(clock, Clock):
            self._clock = clock
        elif callable(clock):
            self._clock = Clock(time_fn=clock)
        else:
            self._clock = Clock()

        # Meta lock guards dictionary of path-level lock objects and fencing tokens.
        self._meta_lock = threading.Lock()
        self._paths: dict[str, _PathLock] = {}
        # Persistent mapping of path -> highest fencing token counter seen across idle cleanups.
        self._fencing_tokens: dict[str, int] = {}

    def _get_path_lock(self, path: str) -> _PathLock:
        """Fetch or create the synchronization object for the given path under the meta lock."""
        with self._meta_lock:
            path_lock = self._paths.get(path)
            if path_lock is None:
                initial_token = self._fencing_tokens.get(path, 0)
                path_lock = _PathLock(path, initial_token=initial_token)
                path_lock.last_active_at = self._clock.now()
                self._paths[path] = path_lock
                # Register condition with clock so FakeClock.advance() wakes waiters deterministically.
                self._clock.register_condition(path_lock.cond)
            return path_lock

    def cleanup_idle(self, max_idle_seconds: float = 0.0) -> int:
        """Evict idle _PathLock objects that have had no holders or waiters for max_idle_seconds."""
        now = self._clock.now()
        removed_count = 0
        with self._meta_lock:
            idle_paths: list[str] = []
            for path, path_lock in list(self._paths.items()):
                with path_lock.cond:
                    path_lock.purge_expired(now)
                    if path_lock.is_idle() and (now - path_lock.last_active_at) >= max_idle_seconds:
                        idle_paths.append(path)

            for path in idle_paths:
                path_lock = self._paths.pop(path, None)
                if path_lock is not None:
                    self._clock.unregister_condition(path_lock.cond)
                    removed_count += 1

        return removed_count

    def acquire(
        self,
        path: str,
        holder_id: str,
        mode: LockMode,
        timeout: float | None = None,
        ttl: float | None = None,
    ) -> Lease:
        """Acquire a shared or exclusive lease on path, blocking until granted or timeout."""
        if mode not in ("shared", "exclusive"):
            raise ValueError(f"Invalid mode '{mode}': must be 'shared' or 'exclusive'")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        actual_ttl = float(ttl) if ttl is not None else self._default_ttl
        if actual_ttl <= 0:
            raise ValueError("ttl must be positive")

        path_lock = self._get_path_lock(path)

        with path_lock.cond:
            now = self._clock.now()
            path_lock.purge_expired(now)

            # Re-entrant acquisition is strictly rejected to prevent lock state corruption.
            if path_lock.exclusive_holder == holder_id:
                raise ValueError(f"Holder '{holder_id}' already holds an exclusive lock on '{path}'")
            if holder_id in path_lock.shared_holders:
                raise ValueError(f"Holder '{holder_id}' already holds a shared lock on '{path}'")

            deadline = (now + timeout) if timeout is not None else None

            if mode == "shared":
                while True:
                    now = self._clock.now()
                    path_lock.purge_expired(now)

                    # Writer preference: queue behind waiting writers to prevent writer starvation.
                    if path_lock.exclusive_holder is None and path_lock.waiting_writers == 0:
                        break

                    if deadline is not None and now >= deadline:
                        raise LockTimeout(f"Timed out acquiring shared lock on '{path}' for '{holder_id}'")

                    wait_timeout = path_lock.calc_wait_timeout(now, deadline)
                    path_lock.cond.wait(timeout=wait_timeout)

                now = self._clock.now()
                expires_at = now + actual_ttl
                path_lock.shared_holders[holder_id] = (expires_at, path_lock.fencing_token)
                path_lock.last_active_at = now
                return Lease(
                    path=path,
                    mode="shared",
                    holder_id=holder_id,
                    token=path_lock.fencing_token,
                    expires_at=expires_at,
                )

            else:  # mode == "exclusive"
                path_lock.waiting_writers += 1
                try:
                    while True:
                        now = self._clock.now()
                        path_lock.purge_expired(now)

                        # Exclusive lock requires zero active exclusive holders and zero shared readers.
                        if path_lock.exclusive_holder is None and len(path_lock.shared_holders) == 0:
                            break

                        if deadline is not None and now >= deadline:
                            raise LockTimeout(
                                f"Timed out acquiring exclusive lock on '{path}' for '{holder_id}'"
                            )

                        wait_timeout = path_lock.calc_wait_timeout(now, deadline)
                        path_lock.cond.wait(timeout=wait_timeout)
                finally:
                    path_lock.waiting_writers -= 1
                    # Wake up blocked readers and other writers so progress resumes immediately.
                    path_lock.cond.notify_all()

                # Monotonically increment fencing token on each exclusive grant.
                path_lock.fencing_token += 1
                now = self._clock.now()
                expires_at = now + actual_ttl
                path_lock.exclusive_holder = holder_id
                path_lock.exclusive_token = path_lock.fencing_token
                path_lock.exclusive_expires_at = expires_at
                path_lock.last_active_at = now
                with self._meta_lock:
                    self._fencing_tokens[path] = path_lock.fencing_token

                return Lease(
                    path=path,
                    mode="exclusive",
                    holder_id=holder_id,
                    token=path_lock.fencing_token,
                    expires_at=expires_at,
                )

    def release(self, lease: Lease) -> None:
        """Release the specified lease. Harmless no-op if lease is expired or unknown."""
        path_lock = self._get_path_lock(lease.path)
        with path_lock.cond:
            now = self._clock.now()
            path_lock.purge_expired(now)

            if lease.mode == "exclusive":
                # Must match exact holder_id AND token AND expiry to release.
                if (
                    path_lock.exclusive_holder == lease.holder_id
                    and path_lock.exclusive_token == lease.token
                    and path_lock.exclusive_expires_at == lease.expires_at
                ):
                    path_lock.exclusive_holder = None
                    path_lock.exclusive_token = 0
                    path_lock.exclusive_expires_at = 0.0
                    path_lock.last_active_at = now
                    # Notify waiters (both waiting writers and queued readers) that resource is free.
                    path_lock.cond.notify_all()
            elif lease.mode == "shared":
                # Must match exact holder_id AND token AND expiry in shared_holders map.
                if (
                    lease.holder_id in path_lock.shared_holders
                    and path_lock.shared_holders[lease.holder_id] == (lease.expires_at, lease.token)
                ):
                    del path_lock.shared_holders[lease.holder_id]
                    path_lock.last_active_at = now
                    # Notify waiters that a shared reader has exited.
                    path_lock.cond.notify_all()

    def renew(self, lease: Lease, ttl: float | None = None) -> Lease:
        """Extend the lease expiry. Raises LeaseExpired if already expired or not held."""
        actual_ttl = float(ttl) if ttl is not None else self._default_ttl
        if actual_ttl <= 0:
            raise ValueError("ttl must be positive")

        path_lock = self._get_path_lock(lease.path)
        with path_lock.cond:
            now = self._clock.now()
            path_lock.purge_expired(now)

            if now >= lease.expires_at:
                raise LeaseExpired(f"Lease for '{lease.path}' held by '{lease.holder_id}' has already expired")

            if lease.mode == "exclusive":
                if (
                    path_lock.exclusive_holder != lease.holder_id
                    or path_lock.exclusive_token != lease.token
                    or path_lock.exclusive_expires_at != lease.expires_at
                ):
                    raise LeaseExpired(
                        f"Exclusive lease on '{lease.path}' is no longer held by '{lease.holder_id}'"
                    )
                new_expires_at = now + actual_ttl
                path_lock.exclusive_expires_at = new_expires_at
                path_lock.last_active_at = now
            else:  # shared
                if (
                    lease.holder_id not in path_lock.shared_holders
                    or path_lock.shared_holders[lease.holder_id] != (lease.expires_at, lease.token)
                ):
                    raise LeaseExpired(
                        f"Shared lease on '{lease.path}' is no longer held by '{lease.holder_id}'"
                    )
                new_expires_at = now + actual_ttl
                path_lock.shared_holders[lease.holder_id] = (new_expires_at, lease.token)
                path_lock.last_active_at = now

            return Lease(
                path=lease.path,
                mode=lease.mode,
                holder_id=lease.holder_id,
                token=lease.token,
                expires_at=new_expires_at,
            )

    def is_valid(self, lease: Lease) -> bool:
        """Return True if lease is currently active and unexpired, purging lazily first."""
        path_lock = self._get_path_lock(lease.path)
        with path_lock.cond:
            now = self._clock.now()
            path_lock.purge_expired(now)

            if now >= lease.expires_at:
                return False

            if lease.mode == "exclusive":
                return (
                    path_lock.exclusive_holder == lease.holder_id
                    and path_lock.exclusive_token == lease.token
                    and path_lock.exclusive_expires_at == lease.expires_at
                )
            else:
                return (
                    lease.holder_id in path_lock.shared_holders
                    and path_lock.shared_holders[lease.holder_id] == (lease.expires_at, lease.token)
                )
