"""Enums for Thermal Status classifications and Transition Event types."""

from enum import Enum


class ThermalStatus(Enum):
    """Classification of processor junction thermal safety status."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    OVERHEATING = "OVERHEATING"


class ThermalTransitionEventType(Enum):
    """Specific transition events occurring when threshold boundaries are crossed."""

    NO_CHANGE = "NO_CHANGE"
    WARNING_ENTERED = "WARNING_ENTERED"
    CRITICAL_ENTERED = "CRITICAL_ENTERED"
    OVERHEATING_ENTERED = "OVERHEATING_ENTERED"
    WARNING_CLEARED = "WARNING_CLEARED"
    CRITICAL_CLEARED = "CRITICAL_CLEARED"
    OVERHEATING_CLEARED = "OVERHEATING_CLEARED"
