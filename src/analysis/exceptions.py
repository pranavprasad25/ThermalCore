"""Exceptions for the ThermoShift Analysis subsystem."""


class AnalysisError(Exception):
    """Base exception for analysis subsystem errors."""

    pass


class InvalidAnalysisError(AnalysisError, ValueError):
    """Raised when simulation result inputs to the analyzer are missing, invalid, or inconsistent."""

    pass
