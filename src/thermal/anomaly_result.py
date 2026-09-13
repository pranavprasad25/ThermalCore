"""Data models for statistical thermal anomaly detection results, status, severity, persistence, and transitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AnomalyStatus(Enum):
    """Status classification of thermal anomaly evaluation."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NORMAL = "NORMAL"
    POSSIBLE_ANOMALY = "POSSIBLE_ANOMALY"
    ANOMALY = "ANOMALY"
    RECOVERED = "RECOVERED"


class AnomalySeverity(Enum):
    """Statistical unusualness severity classification."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyPersistence(Enum):
    """Persistence duration categorization of detected anomaly."""

    NOT_ANOMALOUS = "NOT_ANOMALOUS"
    TEMPORARY = "TEMPORARY"
    PERSISTENT = "PERSISTENT"


class AnomalyTransition(Enum):
    """Transition category between consecutive anomaly evaluations."""

    NO_CHANGE = "NO_CHANGE"
    ANOMALY_STARTED = "ANOMALY_STARTED"
    ANOMALY_ESCALATED = "ANOMALY_ESCALATED"
    ANOMALY_DE_ESCALATED = "ANOMALY_DE_ESCALATED"
    ANOMALY_RECOVERED = "ANOMALY_RECOVERED"


@dataclass(frozen=True)
class ThermalAnomalyResult:
    """Structured result produced by the Thermal Anomaly Detection Engine.

    Attributes:
        timestamp: Timezone-aware UTC datetime of evaluation.
        sensor_id: Identifier string of evaluated sensor.
        current_temperature: Processed/smoothed temperature in °C.
        raw_temperature: Raw acquired temperature in °C.
        baseline_temperature: Computed historical rolling baseline in °C (None if insufficient data).
        deviation: Current temperature minus baseline temperature in °C.
        absolute_deviation: Absolute value of temperature deviation in °C.
        normalized_deviation: Standardized deviation (z-score) relative to variability.
        variability: Standard deviation or dispersion of rolling historical readings in °C.
        anomaly_score: Deterministic score between 0.0 (completely normal) and 100.0 (extremely unusual).
        severity: Severity level (NONE, LOW, MEDIUM, HIGH, CRITICAL).
        status: Detection status (INSUFFICIENT_DATA, NORMAL, POSSIBLE_ANOMALY, ANOMALY, RECOVERED).
        persistence: Anomaly duration status (NOT_ANOMALOUS, TEMPORARY, PERSISTENT).
        transition: Transition classification compared to previous evaluation.
        rate_of_change: Rate of temperature change in °C per second.
        rate_deviation: Difference between current rate of change and historical mean rate.
        reasons: Human-readable explainability evidence lines.
        not_evaluable_reasons: Reasons explaining any unevaluated checks.
    """

    timestamp: datetime
    sensor_id: str
    current_temperature: float
    raw_temperature: float
    baseline_temperature: Optional[float]
    deviation: float
    absolute_deviation: float
    normalized_deviation: float
    variability: float
    anomaly_score: float
    severity: AnomalySeverity
    status: AnomalyStatus
    persistence: AnomalyPersistence
    transition: AnomalyTransition
    rate_of_change: float = 0.0
    rate_deviation: float = 0.0
    reasons: List[str] = field(default_factory=list)
    not_evaluable_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalAnomalyResult to standardized dictionary output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "current_temperature": round(self.current_temperature, 4),
            "raw_temperature": round(self.raw_temperature, 4),
            "baseline_temperature": round(self.baseline_temperature, 4) if self.baseline_temperature is not None else None,
            "deviation": round(self.deviation, 4),
            "absolute_deviation": round(self.absolute_deviation, 4),
            "normalized_deviation": round(self.normalized_deviation, 4),
            "variability": round(self.variability, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity.value,
            "status": self.status.value,
            "persistence": self.persistence.value,
            "transition": self.transition.value,
            "rate_of_change": round(self.rate_of_change, 4),
            "rate_deviation": round(self.rate_deviation, 4),
            "reasons": list(self.reasons),
            "not_evaluable_reasons": list(self.not_evaluable_reasons),
        }
