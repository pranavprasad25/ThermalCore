"""Exceptions for the ThermoShift end-to-end simulation pipeline."""


class SimulationError(Exception):
    """Base exception for all simulation pipeline errors."""

    pass


class InvalidSimulationParameterError(SimulationError, ValueError):
    """Raised when simulation parameters (duration, timestep, initial temp, etc.) are invalid."""

    pass
