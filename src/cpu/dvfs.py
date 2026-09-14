"""DVFS and Workload-to-Operating-Condition Mapping Strategy Implementations."""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.cpu.exceptions import InvalidOperatingConditionError
from src.cpu.operating_condition import OperatingCondition
from src.power.config import PowerConfig


class OperatingConditionMapper(ABC):
    """Abstract base class for mapping normalized CPU workload to operating conditions (voltage, frequency)."""

    @abstractmethod
    def map_workload(self, workload: float) -> OperatingCondition:
        """Map normalized workload [0.0, 1.0] to operating conditions (voltage, frequency).

        Args:
            workload: Normalized CPU workload in [0.0, 1.0].

        Returns:
            OperatingCondition object containing voltage, frequency, and workload.
        """
        pass


class LinearDVFSMapper(OperatingConditionMapper):
    """Linearly maps CPU workload to voltage and frequency between defined minimum and maximum bounds.

    Formula:
        f(w) = min_frequency + workload * (max_frequency - min_frequency)
        v(w) = min_voltage + workload * (max_voltage - min_voltage)
    """

    def __init__(
        self,
        min_voltage: Optional[float] = None,
        max_voltage: Optional[float] = None,
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = None,
        power_config: Optional[PowerConfig] = None,
    ) -> None:
        """Initialize LinearDVFSMapper.

        Args:
            min_voltage: Minimum voltage in Volts.
            max_voltage: Maximum voltage in Volts.
            min_frequency: Minimum frequency in Hz.
            max_frequency: Maximum frequency in Hz.
            power_config: Optional PowerConfig to draw limits from if parameters omitted.
        """
        cfg = power_config if power_config is not None else PowerConfig()

        self.min_voltage = float(min_voltage if min_voltage is not None else cfg.min_voltage)
        self.max_voltage = float(max_voltage if max_voltage is not None else (
            0.8 if max_voltage is None and min_voltage is None and power_config is None else cfg.max_voltage
        ))
        if max_voltage is not None:
            self.max_voltage = float(max_voltage)
        elif power_config is not None:
            self.max_voltage = cfg.max_voltage
        else:
            # Default sensible bounds if no config given
            self.min_voltage = 0.8
            self.max_voltage = 1.2

        self.min_frequency = float(min_frequency if min_frequency is not None else (
            1.0e9 if min_frequency is None and power_config is None else cfg.min_frequency
        ))
        self.max_frequency = float(max_frequency if max_frequency is not None else (
            3.0e9 if max_frequency is None and power_config is None else cfg.max_frequency
        ))

        self._validate_bounds()

    def _validate_bounds(self) -> None:
        if self.min_voltage <= 0 or self.max_voltage <= 0 or self.min_voltage >= self.max_voltage:
            raise InvalidOperatingConditionError(
                f"Invalid voltage bounds: min_voltage ({self.min_voltage}) must be < max_voltage ({self.max_voltage}) and > 0"
            )
        if self.min_frequency <= 0 or self.max_frequency <= 0 or self.min_frequency >= self.max_frequency:
            raise InvalidOperatingConditionError(
                f"Invalid frequency bounds: min_frequency ({self.min_frequency}) must be < max_frequency ({self.max_frequency}) and > 0"
            )

    def map_workload(self, workload: float) -> OperatingCondition:
        """Map normalized workload [0.0, 1.0] to operating conditions."""
        if not isinstance(workload, (int, float)) or not math.isfinite(workload) or not (0.0 <= workload <= 1.0):
            raise InvalidOperatingConditionError(f"Workload must be a float in [0.0, 1.0], got {workload}")

        w = float(workload)
        voltage = self.min_voltage + w * (self.max_voltage - self.min_voltage)
        frequency = self.min_frequency + w * (self.max_frequency - self.min_frequency)

        return OperatingCondition(
            workload=w,
            voltage=voltage,
            frequency=frequency,
            metadata={"mapper": "LinearDVFSMapper"},
        )


class FixedOperatingConditionMapper(OperatingConditionMapper):
    """Maps any workload to a fixed voltage and frequency."""

    def __init__(self, voltage: float = 1.0, frequency: float = 2.5e9) -> None:
        """Initialize fixed mapper.

        Args:
            voltage: Fixed operating voltage in Volts.
            frequency: Fixed operating frequency in Hz.
        """
        if not isinstance(voltage, (int, float)) or not math.isfinite(voltage) or voltage <= 0:
            raise InvalidOperatingConditionError(f"Fixed voltage must be positive, got {voltage}")
        if not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
            raise InvalidOperatingConditionError(f"Fixed frequency must be positive, got {frequency}")

        self.voltage = float(voltage)
        self.frequency = float(frequency)

    def map_workload(self, workload: float) -> OperatingCondition:
        """Return fixed operating conditions for given workload."""
        if not isinstance(workload, (int, float)) or not math.isfinite(workload) or not (0.0 <= workload <= 1.0):
            raise InvalidOperatingConditionError(f"Workload must be in [0.0, 1.0], got {workload}")

        return OperatingCondition(
            workload=float(workload),
            voltage=self.voltage,
            frequency=self.frequency,
            metadata={"mapper": "FixedOperatingConditionMapper"},
        )


class DiscreteDVFSMapper(OperatingConditionMapper):
    """Maps workload to discrete DVFS P-states defined as threshold tuples (min_workload, voltage, frequency)."""

    def __init__(
        self,
        p_states: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> None:
        """Initialize DiscreteDVFSMapper.

        Args:
            p_states: Sequence of (min_workload_threshold, voltage, frequency_hz) sorted by threshold.
                      If omitted, default 4-level P-states are used.
        """
        if p_states is None:
            self._p_states = [
                (0.0, 0.75, 1.0e9),   # Low P-state
                (0.3, 0.90, 1.8e9),   # Mid-low P-state
                (0.6, 1.05, 2.5e9),   # Mid-high P-state
                (0.85, 1.20, 3.2e9),  # High/Turbo P-state
            ]
        else:
            if not p_states:
                raise InvalidOperatingConditionError("p_states sequence cannot be empty")
            self._p_states = list(p_states)

        self._validate_p_states()

    def _validate_p_states(self) -> None:
        for threshold, v, f in self._p_states:
            if not (0.0 <= threshold <= 1.0):
                raise InvalidOperatingConditionError(f"P-state threshold must be in [0.0, 1.0], got {threshold}")
            if v <= 0:
                raise InvalidOperatingConditionError(f"P-state voltage must be positive, got {v}")
            if f <= 0:
                raise InvalidOperatingConditionError(f"P-state frequency must be positive, got {f}")

        # Ensure sorted by threshold
        self._p_states.sort(key=lambda item: item[0])

    def map_workload(self, workload: float) -> OperatingCondition:
        """Find highest applicable P-state for given workload."""
        if not isinstance(workload, (int, float)) or not math.isfinite(workload) or not (0.0 <= workload <= 1.0):
            raise InvalidOperatingConditionError(f"Workload must be in [0.0, 1.0], got {workload}")

        w = float(workload)
        selected_v, selected_f = self._p_states[0][1], self._p_states[0][2]

        for threshold, v, f in self._p_states:
            if w >= threshold:
                selected_v, selected_f = v, f
            else:
                break

        return OperatingCondition(
            workload=w,
            voltage=selected_v,
            frequency=selected_f,
            metadata={"mapper": "DiscreteDVFSMapper"},
        )
