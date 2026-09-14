"""Exceptions for the ThermoShift Workload subsystem."""


class WorkloadError(Exception):
    """Base exception for all workload-related errors."""

    pass


class InvalidWorkloadProfileError(WorkloadError, ValueError):
    """Raised when workload profile parameters, points, or timestamps are invalid."""

    pass
