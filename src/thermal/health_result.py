"""Data models for unified thermal health results and health status enums."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from src.thermal.anomaly_result import AnomalyPersistence, AnomalySeverity, AnomalyStatus, AnomalyTransition
from src.thermal.detection_result import StateTransition, ThermalState
from src.thermal.processed_data import ThermalTrend


class ThermalHealthStatus(Enum):
    """Categorization of overall thermal health status."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ThermalHealthResult:
    """Unified result produced by the Thermal Health Monitoring Engine.

    Attributes:
        timestamp: Timezone-aware UTC datetime of update.
        sensor_id: Identifier string of evaluated sensor.
        temperature: Current processed/smoothed temperature in °C.
        raw_temperature: Raw acquired temperature in °C.
        trend: Thermal trend (HEATING, COOLING, STABLE, UNKNOWN).
        rate_of_change: Rate of temperature change in °C per second.
        thermal_state: Operating thermal state (NORMAL, WARNING, CRITICAL).
        anomaly_status: Anomaly evaluation status (INSUFFICIENT_DATA, NORMAL, POSSIBLE_ANOMALY, ANOMALY, RECOVERED).
        anomaly_severity: Statistical anomaly severity (NONE, LOW, MEDIUM, HIGH, CRITICAL).
        anomaly_score: Deterministic anomaly score in [0.0, 100.0].
        health_score: Overall thermal health score in [0.0, 100.0] (100 = optimal, 0 = critical stress).
        health_status: Overall health status (UNKNOWN, HEALTHY, DEGRADED, WARNING, CRITICAL).
        active_alerts: List of unique active alert/condition strings currently triggering.
        state_transition: Operating thermal state transition from Task 3.
        anomaly_transition: Anomaly state transition from Task 4.
        reasons: Unified human-readable explainability evidence lines.
        penalties: Detailed breakdown of health score penalties.
    """

    timestamp: datetime
    sensor_id: str
    temperature: float
    raw_temperature: float
    trend: ThermalTrend
    rate_of_change: float
    thermal_state: ThermalState
    anomaly_status: AnomalyStatus
    anomaly_severity: AnomalySeverity
    anomaly_score: float
    health_score: float
    health_status: ThermalHealthStatus
    active_alerts: List[str] = field(default_factory=list)
    state_transition: StateTransition = StateTransition.NO_CHANGE
    anomaly_transition: AnomalyTransition = AnomalyTransition.NO_CHANGE
    reasons: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalHealthResult to standardized dictionary output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "temperature": round(self.temperature, 4),
            "raw_temperature": round(self.raw_temperature, 4),
            "trend": self.trend.value,
            "rate_of_change": round(self.rate_of_change, 4),
            "thermal_state": self.thermal_state.value,
            "anomaly_status": self.anomaly_status.value,
            "anomaly_severity": self.anomaly_severity.value,
            "anomaly_score": round(self.anomaly_score, 4),
            "health_score": round(self.health_score, 4),
            "health_status": self.health_status.value,
            "active_alerts": list(self.active_alerts),
            "state_transition": self.state_transition.value,
            "anomaly_transition": self.anomaly_transition.value,
            "reasons": list(self.reasons),
            "penalties": list(self.penalties),
        }
