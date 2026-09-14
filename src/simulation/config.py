"""Simulation Configuration parameters dataclass."""

from dataclasses import dataclass
import math
from typing import Optional

from src.simulation.exceptions import InvalidSimulationParameterError


@dataclass
class SimulationConfig:
    """Centralized configuration for end-to-end workload -> power -> temperature simulation.

    Attributes:
        duration: Total simulation time duration in seconds (> 0).
        timestep: Simulation discretization time step delta_t in seconds (> 0).
        ambient_temperature: Ambient temperature in °C.
        initial_temperature: Starting junction temperature in °C.
        integration_method: Integration scheme for thermal model ('EXACT', 'EULER', 'RK4').
    """

    duration: float = 10.0
    timestep: float = 0.1
    ambient_temperature: Optional[float] = None
    initial_temperature: Optional[float] = None
    integration_method: str = "EXACT"

    def __post_init__(self) -> None:
        """Validate simulation configuration parameters."""
        if not isinstance(self.duration, (int, float)) or not math.isfinite(self.duration) or self.duration <= 0.0:
            raise InvalidSimulationParameterError(
                f"Simulation duration must be a positive finite float in seconds, got {self.duration}"
            )

        if not isinstance(self.timestep, (int, float)) or not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise InvalidSimulationParameterError(
                f"Simulation timestep must be a positive finite float in seconds, got {self.timestep}"
            )

        if self.timestep > self.duration:
            raise InvalidSimulationParameterError(
                f"Timestep ({self.timestep}s) cannot exceed total simulation duration ({self.duration}s)"
            )

        if self.ambient_temperature is not None:
            if not isinstance(self.ambient_temperature, (int, float)) or not math.isfinite(self.ambient_temperature):
                raise InvalidSimulationParameterError(
                    f"Ambient temperature must be a finite float, got {self.ambient_temperature}"
                )
            self.ambient_temperature = float(self.ambient_temperature)

        if self.initial_temperature is not None:
            if not isinstance(self.initial_temperature, (int, float)) or not math.isfinite(self.initial_temperature):
                raise InvalidSimulationParameterError(
                    f"Initial temperature must be a finite float, got {self.initial_temperature}"
                )
            self.initial_temperature = float(self.initial_temperature)

        if not isinstance(self.integration_method, str) or self.integration_method.upper() not in ("EXACT", "EULER", "RK4"):
            raise InvalidSimulationParameterError(
                f"Integration method must be one of ('EXACT', 'EULER', 'RK4'), got '{self.integration_method}'"
            )

        self.duration = float(self.duration)
        self.timestep = float(self.timestep)
        self.integration_method = self.integration_method.upper()
