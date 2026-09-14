"""Data models for thermal safety checking, threshold transitions, and simulation analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.safety.config import ThermalSafetyConfig
from src.safety.status import ThermalStatus, ThermalTransitionEventType


@dataclass(frozen=True)
class ThermalSafetyResult:
    """Outcome of a single-temperature thermal safety check.

    Attributes:
        temperature: Evaluated temperature in °C.
        status: Classified ThermalStatus enum (NORMAL, WARNING, CRITICAL, OVERHEATING).
        is_safe: True ONLY if status is NORMAL.
        is_warning: True if status is WARNING.
        is_critical: True if status is CRITICAL.
        is_overheating: True if status is OVERHEATING.
        timestamp: Optional simulation time in seconds.
        step_index: Optional simulation step index.
    """

    temperature: float
    status: ThermalStatus
    is_safe: bool
    is_warning: bool
    is_critical: bool
    is_overheating: bool
    timestamp: Optional[float] = None
    step_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalSafetyResult to serializable dictionary format."""
        return {
            "temperature": round(self.temperature, 4),
            "status": self.status.value,
            "is_safe": self.is_safe,
            "is_warning": self.is_warning,
            "is_critical": self.is_critical,
            "is_overheating": self.is_overheating,
            "timestamp": round(self.timestamp, 4) if self.timestamp is not None else None,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class ThermalTransitionEvent:
    """Represents a state transition event when temperature crosses a threshold boundary.

    Attributes:
        step_index: Simulation step index where transition occurred.
        time: Timestamp in seconds of transition event.
        temperature: Temperature in °C at transition point.
        previous_status: Prior ThermalStatus.
        new_status: Updated ThermalStatus.
        event_type: ThermalTransitionEventType enum.
    """

    step_index: int
    time: float
    temperature: float
    previous_status: ThermalStatus
    new_status: ThermalStatus
    event_type: ThermalTransitionEventType

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition event to serializable dictionary."""
        return {
            "step_index": self.step_index,
            "time": round(self.time, 4),
            "temperature": round(self.temperature, 4),
            "previous_status": self.previous_status.value,
            "new_status": self.new_status.value,
            "event_type": self.event_type.value,
        }


@dataclass(frozen=True)
class ThermalSafetyAnalysis:
    """Aggregated safety analysis for a complete simulation time series.

    Attributes:
        maximum_temperature: Peak temperature recorded in °C.
        time_of_maximum_temperature: Timestamp of peak temperature in seconds.
        final_temperature: Temperature at simulation end in °C.
        total_timesteps: Total number of evaluated timesteps.
        duration: Total simulation time duration in seconds.
        normal_timesteps: Number of timesteps in NORMAL state.
        warning_timesteps: Number of timesteps in WARNING state.
        critical_timesteps: Number of timesteps in CRITICAL state.
        overheating_timesteps: Number of timesteps in OVERHEATING state.
        normal_duration: Total duration spent in NORMAL state in seconds.
        warning_duration: Total duration spent in WARNING state in seconds.
        critical_duration: Total duration spent in CRITICAL state in seconds.
        overheating_duration: Total duration spent in OVERHEATING state in seconds.
        has_violations: True if any timestep departed from NORMAL state.
        has_overheating: True if any timestep reached OVERHEATING state.
        first_warning_time: Timestamp of first WARNING transition, if any.
        first_critical_time: Timestamp of first CRITICAL transition, if any.
        first_overheating_time: Timestamp of first OVERHEATING transition, if any.
        transitions: List of detected ThermalTransitionEvent objects.
        step_results: List of individual ThermalSafetyResult objects for each step.
        config: ThermalSafetyConfig used for monitoring analysis.
    """

    maximum_temperature: float
    time_of_maximum_temperature: float
    final_temperature: float
    total_timesteps: int
    duration: float
    normal_timesteps: int
    warning_timesteps: int
    critical_timesteps: int
    overheating_timesteps: int
    normal_duration: float
    warning_duration: float
    critical_duration: float
    overheating_duration: float
    has_violations: bool
    has_overheating: bool
    first_warning_time: Optional[float]
    first_critical_time: Optional[float]
    first_overheating_time: Optional[float]
    transitions: List[ThermalTransitionEvent]
    step_results: List[ThermalSafetyResult]
    config: ThermalSafetyConfig

    def summary(self) -> Dict[str, Any]:
        """Return a structured summary dictionary of thermal safety analysis."""
        return {
            "maximum_temperature": round(self.maximum_temperature, 4),
            "time_of_maximum_temperature": round(self.time_of_maximum_temperature, 4),
            "final_temperature": round(self.final_temperature, 4),
            "total_timesteps": self.total_timesteps,
            "duration": round(self.duration, 4),
            "has_violations": self.has_violations,
            "has_overheating": self.has_overheating,
            "first_warning_time": round(self.first_warning_time, 4) if self.first_warning_time is not None else None,
            "first_critical_time": round(self.first_critical_time, 4) if self.first_critical_time is not None else None,
            "first_overheating_time": round(self.first_overheating_time, 4) if self.first_overheating_time is not None else None,
            "normal_timesteps": self.normal_timesteps,
            "warning_timesteps": self.warning_timesteps,
            "critical_timesteps": self.critical_timesteps,
            "overheating_timesteps": self.overheating_timesteps,
            "normal_duration_seconds": round(self.normal_duration, 4),
            "warning_duration_seconds": round(self.warning_duration, 4),
            "critical_duration_seconds": round(self.critical_duration, 4),
            "overheating_duration_seconds": round(self.overheating_duration, 4),
            "total_transitions": len(self.transitions),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalSafetyAnalysis to serializable dictionary."""
        return {
            "summary": self.summary(),
            "config": {
                "warning_temperature": self.config.warning_temperature,
                "critical_temperature": self.config.critical_temperature,
                "maximum_temperature": self.config.maximum_temperature,
                "hysteresis": self.config.hysteresis,
            },
            "transitions": [t.to_dict() for t in self.transitions],
            "step_results": [s.to_dict() for s in self.step_results],
        }
