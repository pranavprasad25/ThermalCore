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


class ThermalModelError(Exception):
    """Base exception for thermal RC model and simulation errors."""
    pass


class InvalidThermalParameterError(ThermalModelError, ValueError):
    """Raised when a thermal physical parameter, timestep, or input power is invalid, non-numeric, or out of bounds."""
    pass


class ThermalSimulationError(ThermalModelError):
    """Raised when thermal simulation integration fails or encounters numerical anomalies."""
    pass

