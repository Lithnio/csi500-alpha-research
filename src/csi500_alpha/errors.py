class Csi500AlphaError(Exception):
    """Base exception for expected project failures."""


class ConfigurationError(Csi500AlphaError):
    """Raised when configuration is missing or internally inconsistent."""


class CredentialError(Csi500AlphaError):
    """Raised when a required credential is unavailable."""


class DataFetchError(Csi500AlphaError):
    """Raised when a vendor request cannot be completed safely."""


class DataQualityError(Csi500AlphaError):
    """Raised when a critical data-quality gate fails."""


class InsufficientTrainingData(Csi500AlphaError):
    """Raised when a model cannot yet be fitted without violating its contract."""
