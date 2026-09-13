"""Domain exceptions for thermal data acquisition and validation."""


class ThermalAcquisitionError(Exception):
    """Base exception for all thermal acquisition errors."""
    pass


class InvalidReadingError(ThermalAcquisitionError):
    """Raised when an incoming temperature reading is malformed, non-numeric, infinite, or invalid."""
    pass


class SensorReadError(ThermalAcquisitionError):
    """Raised when a sensor source fails to return a valid payload or encounters an error."""
    pass


class UnsupportedUnitError(InvalidReadingError):
    """Raised when an unsupported temperature unit is specified."""
    pass
