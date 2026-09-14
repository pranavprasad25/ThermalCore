"""Exceptions for the ThermoShift Thermal Safety subsystem."""


class ThermalSafetyError(Exception):
    """Base exception for thermal safety subsystem errors."""

    pass


class InvalidThermalSafetyParameterError(ThermalSafetyError, ValueError):
    """Raised when safety configuration parameters, thresholds, or temperature inputs are invalid."""

    pass
