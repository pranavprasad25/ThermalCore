"""Exceptions for CPU operating condition models and DVFS mappers."""


class OperatingConditionError(Exception):
    """Base exception for operating condition errors."""

    pass


class InvalidOperatingConditionError(OperatingConditionError, ValueError):
    """Raised when operating condition parameters or DVFS states are invalid."""

    pass
