"""Data models for processed thermal statistics and trend analysis."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from src.thermal.reading import ThermalReading


class ThermalTrend(Enum):
    """Thermal trend categorization based on temperature rate of change."""

    HEATING = "HEATING"
    COOLING = "COOLING"
    STABLE = "STABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProcessedThermalData:
    """Standardized representation of processed thermal metrics derived from raw readings.

    Attributes:
        raw_reading: Original raw ThermalReading (preserved intact).
        processed_temperature: Filtered/smoothed temperature in °C.
        moving_average: Rolling moving average temperature in °C over configured window.
        rate_of_change: Rate of temperature change in °C per second (ΔT / Δt).
        trend: ThermalTrend categorization (HEATING, COOLING, STABLE, UNKNOWN).
    """

    raw_reading: ThermalReading
    processed_temperature: float
    moving_average: float
    rate_of_change: float
    trend: ThermalTrend

    def __post_init__(self) -> None:
        """Coerce numeric values."""
        object.__setattr__(self, "processed_temperature", float(self.processed_temperature))
        object.__setattr__(self, "moving_average", float(self.moving_average))
        object.__setattr__(self, "rate_of_change", float(self.rate_of_change))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessedThermalData to dictionary format."""
        return {
            "current_temperature": round(self.processed_temperature, 4),
            "raw_temperature": round(self.raw_reading.temperature, 4),
            "sensor_id": self.raw_reading.sensor_id,
            "timestamp": self.raw_reading.timestamp.isoformat(),
            "moving_average": round(self.moving_average, 4),
            "rate_of_change": round(self.rate_of_change, 4),
            "trend": self.trend.value,
        }
