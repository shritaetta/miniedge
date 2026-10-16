"""Domain exceptions for MiniEdge operations and concurrency control."""


class MiniEdgeError(Exception):
    """Base exception for all MiniEdge errors."""


class Throttled(MiniEdgeError):
    """Raised when the backend rejects a request due to rate limit throttling."""


class TransientBackendError(MiniEdgeError):
    """Raised when an object store operation encounters a temporary recoverable failure."""


class VersionConflict(MiniEdgeError):
    """Raised when an object store compare-and-swap operation detects a version mismatch."""


class StaleFencingToken(MiniEdgeError):
    """Raised when an operation provides a fencing token lower than the highest seen token."""


class LeaseExpired(MiniEdgeError):
    """Raised when attempting to act on or renew an expired lease."""


class LockTimeout(MiniEdgeError):
    """Raised when acquiring a lock exceeds the caller's timeout limit."""


class RateLimited(MiniEdgeError):
    """Raised when a request is rejected by a client or backend token bucket."""
