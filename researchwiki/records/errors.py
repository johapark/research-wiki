"""Archive failures are separate from invalid caller requests and paid retries."""

from ..errors import EnvironmentFailure


class RecordStoreUnavailable(EnvironmentFailure):
    """Storage cannot complete an operation; no provider retry is implied."""


class RecordCorrupt(RecordStoreUnavailable):
    """Existing bytes or identities are invalid; preserve them for inspection."""


class CheckpointStale(RecordStoreUnavailable):
    """A valid event history has a missing or older, explicitly repairable checkpoint."""


class RecordConflict(ValueError):
    """A requested change contradicts an immutable record or closed attempt."""
