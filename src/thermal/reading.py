"""Thermal reading data model, validation, and unit normalization."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Union
from src.thermal.exceptions import InvalidReadingError, UnsupportedUnitError

SUPPORTED_UNITS = {"C", "CELSIUS", "F", "FAHRENHEIT", "K", "KELVIN"}


def validate_temperature_value(value: Any) -> float:
    """Validate that temperature is a finite numeric value.

    Raises:
        InvalidReadingError: If value is None, boolean, non-numeric, NaN, or infinite.
    """
    if value is None:
        raise InvalidReadingError("Temperature value cannot be None.")

    if isinstance(value, bool):
        raise InvalidReadingError(f"Temperature value cannot be boolean: {value}")

    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (ValueError, TypeError) as err:
            raise InvalidReadingError(f"Temperature value must be numeric, got {type(value).__name__}: {value}") from err

    float_val = float(value)
    if not math.isfinite(float_val):
        raise InvalidReadingError(f"Temperature value must be finite (got {float_val}).")

    return float_val


def validate_sensor_id(sensor_id: Any) -> str:
    """Validate that sensor_id is a non-empty string.

    Raises:
        InvalidReadingError: If sensor_id is invalid or empty.
    """
    if not sensor_id or not isinstance(sensor_id, str) or not sensor_id.strip():
        raise InvalidReadingError(f"Sensor ID must be a non-empty string, got: {sensor_id}")
    return sensor_id.strip()


def validate_and_normalize_unit(unit: Any) -> str:
    """Validate temperature unit string.

    Raises:
        UnsupportedUnitError: If unit is unsupported or invalid.
    """
    if not unit or not isinstance(unit, str):
        raise UnsupportedUnitError(f"Invalid unit specification: {unit}")

    clean_unit = unit.strip().upper()
    if clean_unit not in SUPPORTED_UNITS:
        raise UnsupportedUnitError(f"Unsupported unit '{unit}'. Supported units: {sorted(list(SUPPORTED_UNITS))}")

    if clean_unit in {"C", "CELSIUS"}:
        return "C"
    elif clean_unit in {"F", "FAHRENHEIT"}:
        return "F"
    elif clean_unit in {"K", "KELVIN"}:
        return "K"

    return "C"


def normalize_to_celsius(value: float, unit: str) -> float:
    """Convert temperature value from specified unit to Celsius (°C)."""
    norm_unit = validate_and_normalize_unit(unit)
    if norm_unit == "C":
        return float(value)
    elif norm_unit == "F":
        return float((value - 32.0) * (5.0 / 9.0))
    elif norm_unit == "K":
        return float(value - 273.15)
    return float(value)


def validate_and_parse_timestamp(ts: Optional[Union[datetime, str, int, float]] = None) -> datetime:
    """Parse and validate input timestamp, returning a timezone-aware UTC datetime object.

    If ts is None, auto-generates current UTC datetime.
    """
    if ts is None:
        return datetime.now(timezone.utc)

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    if isinstance(ts, (int, float)):
        if not math.isfinite(float(ts)) or ts < 0:
            raise InvalidReadingError(f"Invalid epoch timestamp: {ts}")
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as err:
            raise InvalidReadingError(f"Timestamp out of bounds: {ts}") from err

    if isinstance(ts, str):
        clean_str = ts.strip()
        if not clean_str:
            raise InvalidReadingError("Timestamp string cannot be empty.")
        try:
            parsed = datetime.fromisoformat(clean_str)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError as err:
            raise InvalidReadingError(f"Invalid ISO timestamp string '{ts}'") from err

    raise InvalidReadingError(f"Unsupported timestamp format type {type(ts).__name__}")


@dataclass(frozen=True)
class ThermalReading:
    """Standardized internal representation of a thermal sensor reading.

    Attributes:
        temperature: Normalized temperature value in Celsius (°C).
        timestamp: Timezone-aware UTC datetime of reading acquisition.
        sensor_id: Identifier string of sensor source.
        unit: Standardized internal unit string (always "C").
        raw_temperature: Original input temperature before normalization.
        original_unit: Original input unit before normalization.
    """

    temperature: float
    timestamp: datetime
    sensor_id: str
    unit: str = "C"
    raw_temperature: float = 0.0
    original_unit: str = "C"

    def __post_init__(self) -> None:
        """Enforce immutability validation on creation."""
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "raw_temperature", float(self.raw_temperature))

    def to_dict(self) -> Dict[str, Any]:
        """Return standardized dictionary representation."""
        return {
            "temperature": round(self.temperature, 4),
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "unit": self.unit,
            "raw_temperature": round(self.raw_temperature, 4),
            "original_unit": self.original_unit,
        }
