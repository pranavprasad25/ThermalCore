"""Data models for threshold detection results, conditions, and state transitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ThermalState(Enum):
    """Operating thermal state of a CPU sensor/core based on threshold bounds."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"



class ThermalCondition(Enum):
    """Specific thermal conditions recognized by the deterministic rule engine."""

    OVERHEATING = "OVERHEATING"
    SUDDEN_SPIKE = "SUDDEN_SPIKE"
    RAPID_RISE = "RAPID_RISE"
    ABNORMAL_COOLING = "ABNORMAL_COOLING"
    SUSTAINED_HIGH = "SUSTAINED_HIGH"
    RECOVERY = "RECOVERY"


class StateTransition(Enum):
    """Categorization of thermal state transitions between consecutive evaluations."""

    NO_CHANGE = "NO_CHANGE"
    ESCALATED = "ESCALATED"
    DE_ESCALATED = "DE_ESCALATED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class ThermalDetectionResult:
    """Structured result produced by the Thermal Threshold & Rule Detection Engine.

    Attributes:
        timestamp: Timezone-aware UTC datetime of evaluation.
        sensor_id: Identifier string of evaluated sensor.
        temperature: Current processed/smoothed temperature in °C.
        raw_temperature: Raw acquired temperature in °C.
        state: Operating thermal state (NORMAL, WARM/WARNING, HOT/CRITICAL).
        previous_state: Previous operating state before evaluation (if tracked).
        conditions: List of detected deterministic ThermalConditions.
        transition: StateTransition classification (NO_CHANGE, ESCALATED, DE_ESCALATED, RECOVERED).
        rate_of_change: Rate of temperature change in °C per second.
        reasons: Evidence and reasons explaining triggered state and conditions.
        not_evaluable_reasons: Reasons explaining any rules that could not be evaluated due to insufficient data.
    """

    timestamp: datetime
    sensor_id: str
    temperature: float
    raw_temperature: float
    state: ThermalState
    previous_state: Optional[ThermalState]
    conditions: List[ThermalCondition] = field(default_factory=list)
    transition: StateTransition = StateTransition.NO_CHANGE
    rate_of_change: float = 0.0
    reasons: List[str] = field(default_factory=list)
    not_evaluable_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalDetectionResult to standardized dictionary output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "temperature": round(self.temperature, 4),
            "raw_temperature": round(self.raw_temperature, 4),
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "conditions": [c.value for c in self.conditions],
            "transition": self.transition.value,
            "rate_of_change": round(self.rate_of_change, 4),
            "reasons": list(self.reasons),
            "not_evaluable_reasons": list(self.not_evaluable_reasons),
        }
