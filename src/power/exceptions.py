"""Domain exceptions for power estimation and parameter validation."""


class PowerError(Exception):
    """Base exception for all power estimation errors."""
    pass


class InvalidPowerParameterError(PowerError, ValueError):
    """Raised when an operating condition or configuration parameter is invalid, non-numeric, or out of bounds."""
    pass


class PowerEstimationError(PowerError):
    """Raised when power calculation fails or encounters an unrecoverable mathematical error."""
    pass
