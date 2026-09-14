"""Structured results and data representations for thermal RC simulation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class ThermalStepResult:
    """Represents the thermal state at a discrete simulation step.

    Attributes:
        time: Elapsed simulation time in seconds.
        temperature: Current junction/core temperature in °C.
        power: Dissipated power applied during this step in Watts (W).
        temperature_rise: Temperature elevation above ambient in °C (T - Tambient).
    """

    time: float
    temperature: float
    power: float
    temperature_rise: float

    def __post_init__(self) -> None:
        """Coerce numeric values."""
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "power", float(self.power))
        object.__setattr__(self, "temperature_rise", float(self.temperature_rise))

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalStepResult to serializable dictionary."""
        return {
            "time": round(self.time, 4),
            "temperature": round(self.temperature, 4),
            "power": round(self.power, 4),
            "temperature_rise": round(self.temperature_rise, 4),
        }


@dataclass(frozen=True)
class ThermalSimulationResult:
    """Comprehensive structured outcome of a transient thermal RC simulation.

    Attributes:
        times: List of simulation time points in seconds.
        temperatures: List of simulated temperatures in °C at each time point.
        powers: List of power values in Watts at each time point.
        steps: List of ThermalStepResult objects for each simulation step.
        ambient_temperature: Ambient temperature in °C during simulation.
        initial_temperature: Starting temperature in °C.
        thermal_resistance: Thermal resistance R_th in °C/W.
        thermal_capacitance: Thermal capacitance C_th in J/°C.
        time_constant: Thermal time constant tau = R_th * C_th in seconds.
        duration: Total simulation duration in seconds.
        timestep: Time step delta_t in seconds.
    """

    times: List[float]
    temperatures: List[float]
    powers: List[float]
    steps: List[ThermalStepResult]
    ambient_temperature: float
    initial_temperature: float
    thermal_resistance: float
    thermal_capacitance: float
    time_constant: float
    duration: float
    timestep: float

    @property
    def final_temperature(self) -> float:
        """Final recorded temperature at the end of simulation in °C."""
        return self.temperatures[-1] if self.temperatures else self.initial_temperature

    @property
    def max_temperature(self) -> float:
        """Peak temperature reached during simulation in °C."""
        return max(self.temperatures) if self.temperatures else self.initial_temperature

    @property
    def min_temperature(self) -> float:
        """Lowest temperature reached during simulation in °C."""
        return min(self.temperatures) if self.temperatures else self.initial_temperature

    @property
    def final_power(self) -> float:
        """Power at the final time point in Watts."""
        return self.powers[-1] if self.powers else 0.0

    @property
    def steady_state_temperature(self) -> float:
        """Theoretical steady-state temperature for the final power level in °C."""
        return self.ambient_temperature + (self.final_power * self.thermal_resistance)

    @property
    def final_temperature_rise(self) -> float:
        """Final temperature elevation above ambient in °C."""
        return self.final_temperature - self.ambient_temperature

    def __len__(self) -> int:
        """Number of simulation steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[ThermalStepResult]:
        """Iterate over individual simulation steps."""
        return iter(self.steps)

    def __getitem__(self, index: int) -> ThermalStepResult:
        """Index access to simulation steps."""
        return self.steps[index]

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalSimulationResult to serializable dictionary."""
        return {
            "ambient_temperature": round(self.ambient_temperature, 2),
            "initial_temperature": round(self.initial_temperature, 2),
            "thermal_resistance": round(self.thermal_resistance, 4),
            "thermal_capacitance": round(self.thermal_capacitance, 4),
            "time_constant": round(self.time_constant, 4),
            "duration": round(self.duration, 4),
            "timestep": round(self.timestep, 4),
            "total_steps": len(self.steps),
            "final_temperature": round(self.final_temperature, 4),
            "max_temperature": round(self.max_temperature, 4),
            "min_temperature": round(self.min_temperature, 4),
            "steady_state_temperature": round(self.steady_state_temperature, 4),
            "final_temperature_rise": round(self.final_temperature_rise, 4),
            "steps": [s.to_dict() for s in self.steps],
        }
