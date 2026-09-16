"""Exceptions for the ThermoShift Visualization subsystem."""


class VisualizationError(Exception):
    """Base exception for visualization subsystem errors."""

    pass


class InvalidVisualizationError(VisualizationError, ValueError):
    """Raised when visualization input data or parameters are invalid."""

    pass
