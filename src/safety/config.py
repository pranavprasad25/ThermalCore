"""Configuration dataclass for thermal safety limits and thresholds."""

from dataclasses import dataclass
import math
from typing import Optional

from src.safety.exceptions import InvalidThermalSafetyParameterError


@dataclass
class ThermalSafetyConfig:
    """Centralized configuration for thermal safety thresholds and monitoring limits.

    Hierarchy:
        T < warning_temperature                    -> NORMAL
        warning_temperature <= T < critical_temp  -> WARNING
        critical_temp <= T < maximum_temperature    -> CRITICAL
        T >= maximum_temperature                   -> OVERHEATING

    Attributes:
        warning_temperature: Temperature threshold in °C triggering WARNING state.
        critical_temperature: Temperature threshold in °C triggering CRITICAL state.
        maximum_temperature: Temperature threshold in °C triggering OVERHEATING state (overheating boundary).
        hysteresis: Temperature offset in °C required to de-escalate states to prevent flapping.
    """

    warning_temperature: float = 70.0
    critical_temperature: float = 85.0
    maximum_temperature: float = 95.0
    hysteresis: float = 0.0

    def __post_init__(self) -> None:
        """Validate thermal limit ordering and parameter physical sanity."""
        for name, val in [
            ("warning_temperature", self.warning_temperature),
            ("critical_temperature", self.critical_temperature),
            ("maximum_temperature", self.maximum_temperature),
            ("hysteresis", self.hysteresis),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidThermalSafetyParameterError(
                    f"Thermal safety parameter '{name}' must be a finite float, got {val}"
                )

        if self.hysteresis < 0.0:
            raise InvalidThermalSafetyParameterError(
                f"Hysteresis must be a non-negative float, got {self.hysteresis}"
            )

        if self.warning_temperature >= self.critical_temperature:
            raise InvalidThermalSafetyParameterError(
                f"warning_temperature ({self.warning_temperature}°C) must be strictly less than critical_temperature ({self.critical_temperature}°C)"
            )

        if self.critical_temperature >= self.maximum_temperature:
            raise InvalidThermalSafetyParameterError(
                f"critical_temperature ({self.critical_temperature}°C) must be strictly less than maximum_temperature ({self.maximum_temperature}°C)"
            )

        self.warning_temperature = float(self.warning_temperature)
        self.critical_temperature = float(self.critical_temperature)
        self.maximum_temperature = float(self.maximum_temperature)
        self.hysteresis = float(self.hysteresis)

    @property
    def overheating_temperature(self) -> float:
        """Alias for maximum_temperature in °C."""
        return self.maximum_temperature
