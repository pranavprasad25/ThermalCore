"""Configuration dataclass for the Power Estimation subsystem."""

from dataclasses import dataclass
import math
from src.power.exceptions import InvalidPowerParameterError


@dataclass
class PowerConfig:
    """Centralized configuration for power estimation parameters and operating constraints.

    Attributes:
        capacitance: Effective switching capacitance (C_eff) in Farads (F). Default is 1.5e-9 F (1.5 nF).
        base_static_power: Static/leakage baseline power consumption at reference conditions in Watts (W).
        activity_factor_scale: Activity factor scaling multiplier applied to normalized workload [0.0 - 1.0].
        base_activity_factor: Base activity factor at zero workload (idle activity).
        min_voltage: Minimum allowable operating voltage in Volts (V).
        max_voltage: Maximum allowable operating voltage in Volts (V).
        min_frequency: Minimum allowable operating frequency in Hertz (Hz).
        max_frequency: Maximum allowable operating frequency in Hertz (Hz).
        reference_voltage: Nominal reference voltage for voltage-dependent leakage scaling in Volts (V).
        reference_temperature: Reference temperature for thermal leakage scaling in °C.
        temperature_coefficient: Temperature coefficient for leakage scaling in 1/°C.
        static_power_model: Static power model type ("CONSTANT", "LINEAR", or "EXPONENTIAL").
        voltage_dependent_leakage: Whether static power scales with operating voltage relative to reference voltage.
    """

    capacitance: float = 1.5e-9
    base_static_power: float = 5.0
    activity_factor_scale: float = 1.0
    base_activity_factor: float = 0.0
    min_voltage: float = 0.1
    max_voltage: float = 2.5
    min_frequency: float = 1.0e6
    max_frequency: float = 1.0e11
    reference_voltage: float = 1.0
    reference_temperature: float = 25.0
    temperature_coefficient: float = 0.015
    static_power_model: str = "CONSTANT"
    voltage_dependent_leakage: bool = False

    def __post_init__(self) -> None:
        """Validate physical constraints and parameter integrity."""
        # Validate capacitance
        if not isinstance(self.capacitance, (int, float)) or not math.isfinite(self.capacitance) or self.capacitance <= 0:
            raise InvalidPowerParameterError(
                f"Capacitance must be a positive finite float in Farads, got {self.capacitance}"
            )

        # Validate base static power
        if not isinstance(self.base_static_power, (int, float)) or not math.isfinite(self.base_static_power) or self.base_static_power < 0:
            raise InvalidPowerParameterError(
                f"Base static power must be a non-negative finite float in Watts, got {self.base_static_power}"
            )

        # Validate activity factor scaling
        if not isinstance(self.activity_factor_scale, (int, float)) or not math.isfinite(self.activity_factor_scale) or self.activity_factor_scale < 0:
            raise InvalidPowerParameterError(
                f"activity_factor_scale must be non-negative, got {self.activity_factor_scale}"
            )

        # Validate base activity factor
        if not isinstance(self.base_activity_factor, (int, float)) or not math.isfinite(self.base_activity_factor) or self.base_activity_factor < 0:
            raise InvalidPowerParameterError(
                f"base_activity_factor must be non-negative, got {self.base_activity_factor}"
            )

        # Validate voltage limits
        if not isinstance(self.min_voltage, (int, float)) or not math.isfinite(self.min_voltage) or self.min_voltage <= 0:
            raise InvalidPowerParameterError(
                f"min_voltage must be a positive finite float, got {self.min_voltage}"
            )
        if not isinstance(self.max_voltage, (int, float)) or not math.isfinite(self.max_voltage) or self.max_voltage <= 0:
            raise InvalidPowerParameterError(
                f"max_voltage must be a positive finite float, got {self.max_voltage}"
            )
        if self.min_voltage >= self.max_voltage:
            raise InvalidPowerParameterError(
                f"min_voltage ({self.min_voltage}) must be strictly less than max_voltage ({self.max_voltage})"
            )

        # Validate frequency limits
        if not isinstance(self.min_frequency, (int, float)) or not math.isfinite(self.min_frequency) or self.min_frequency <= 0:
            raise InvalidPowerParameterError(
                f"min_frequency must be a positive finite float in Hz, got {self.min_frequency}"
            )
        if not isinstance(self.max_frequency, (int, float)) or not math.isfinite(self.max_frequency) or self.max_frequency <= 0:
            raise InvalidPowerParameterError(
                f"max_frequency must be a positive finite float in Hz, got {self.max_frequency}"
            )
        if self.min_frequency >= self.max_frequency:
            raise InvalidPowerParameterError(
                f"min_frequency ({self.min_frequency}) must be strictly less than max_frequency ({self.max_frequency})"
            )

        # Validate reference voltage & temperature
        if not isinstance(self.reference_voltage, (int, float)) or not math.isfinite(self.reference_voltage) or self.reference_voltage <= 0:
            raise InvalidPowerParameterError(
                f"reference_voltage must be a positive finite float, got {self.reference_voltage}"
            )
        if not isinstance(self.reference_temperature, (int, float)) or not math.isfinite(self.reference_temperature):
            raise InvalidPowerParameterError(
                f"reference_temperature must be a finite float in °C, got {self.reference_temperature}"
            )

        # Validate temperature coefficient
        if not isinstance(self.temperature_coefficient, (int, float)) or not math.isfinite(self.temperature_coefficient) or self.temperature_coefficient < 0:
            raise InvalidPowerParameterError(
                f"temperature_coefficient must be non-negative, got {self.temperature_coefficient}"
            )

        # Validate static power model enum
        if not isinstance(self.static_power_model, str) or self.static_power_model.upper() not in ("CONSTANT", "LINEAR", "EXPONENTIAL"):
            raise InvalidPowerParameterError(
                f"static_power_model must be one of ('CONSTANT', 'LINEAR', 'EXPONENTIAL'), got '{self.static_power_model}'"
            )
        self.static_power_model = self.static_power_model.upper()
