"""Per-Core Temperature Monitoring & History System for ThermoShift."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from src.thermal.exceptions import InvalidReadingError


@dataclass
class CoreThermalState:
    """Per-core thermal state tracking current/previous readings, deltas, and bounded history.

    Attributes:
        core_id: Identifier of the CPU core (int or str).
        current_temperature: Latest temperature reading in Celsius (°C).
        previous_temperature: Previous temperature reading in Celsius (°C), or None if first reading.
        temperature_difference: Delta T (current - previous) in °C, or None if first reading.
        current_timestamp: Timestamp of the latest reading (seconds float or datetime).
        previous_timestamp: Timestamp of the previous reading, or None if first reading.
        time_difference: Delta t in seconds between current and previous reading, or None.
        history: Bounded list of (timestamp, temperature) tuples.
    """

    core_id: Union[int, str]
    current_temperature: float
    previous_temperature: Optional[float] = None
    temperature_difference: Optional[float] = None
    current_timestamp: Optional[Union[float, int, datetime]] = None
    previous_timestamp: Optional[Union[float, int, datetime]] = None
    time_difference: Optional[float] = None
    history: List[Tuple[Any, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return structured dictionary representation of core thermal state."""
        return {
            "core_id": self.core_id,
            "current_temperature": round(self.current_temperature, 4) if self.current_temperature is not None else None,
            "previous_temperature": round(self.previous_temperature, 4) if self.previous_temperature is not None else None,
            "temperature_difference": round(self.temperature_difference, 4) if self.temperature_difference is not None else None,
            "current_timestamp": self.current_timestamp.isoformat() if isinstance(self.current_timestamp, datetime) else self.current_timestamp,
            "previous_timestamp": self.previous_timestamp.isoformat() if isinstance(self.previous_timestamp, datetime) else self.previous_timestamp,
            "time_difference": round(self.time_difference, 4) if self.time_difference is not None else None,
            "history_length": len(self.history),
        }


class TemperatureMonitor:
    """Per-core temperature monitoring subsystem for multi-core CPU architectures.

    Tracks temperature readings per core independently, calculates temperature differences (delta T)
    and sampling time deltas (delta t), maintains bounded history buffers, and validates input integrity.
    """

    def __init__(
        self,
        history_size: int = 100,
        min_valid_temp: float = -50.0,
        max_valid_temp: float = 150.0,
    ) -> None:
        """Initialize per-core temperature monitor.

        Args:
            history_size: Configurable maximum number of recent readings to preserve per core (default 100).
            min_valid_temp: Minimum valid physical Celsius temperature bound (default -50.0 °C).
            max_valid_temp: Maximum valid physical Celsius temperature bound (default 150.0 °C).

        Raises:
            InvalidReadingError: If configuration parameters are invalid.
        """
        if not isinstance(history_size, int) or history_size <= 0:
            raise InvalidReadingError(f"history_size must be a positive integer, got {history_size}")

        if not isinstance(min_valid_temp, (int, float)) or not math.isfinite(min_valid_temp):
            raise InvalidReadingError(f"min_valid_temp must be a finite float, got {min_valid_temp}")

        if not isinstance(max_valid_temp, (int, float)) or not math.isfinite(max_valid_temp):
            raise InvalidReadingError(f"max_valid_temp must be a finite float, got {max_valid_temp}")

        if min_valid_temp >= max_valid_temp:
            raise InvalidReadingError(
                f"min_valid_temp ({min_valid_temp}) must be strictly less than max_valid_temp ({max_valid_temp})"
            )

        self.history_size: int = history_size
        self.min_valid_temp: float = float(min_valid_temp)
        self.max_valid_temp: float = float(max_valid_temp)

        # Internal state maps: core_id -> state and history deque
        self._states: Dict[Union[int, str], CoreThermalState] = {}
        self._history_buffers: Dict[Union[int, str], deque[Tuple[Any, float]]] = {}

    def update_temperature(
        self,
        core_id: Union[int, str],
        temperature: float,
        timestamp: Optional[Union[float, int, datetime]] = None,
    ) -> CoreThermalState:
        """Update thermal state for a given CPU core with a new temperature measurement.

        Args:
            core_id: Target CPU core identifier (int or str).
            temperature: Core temperature measurement in Celsius (°C).
            timestamp: Optional timestamp of reading (seconds float/int or datetime).

        Returns:
            Updated CoreThermalState dataclass instance for the core.

        Raises:
            InvalidReadingError: If core_id, temperature, or timestamp fails validation.
        """
        valid_core_id = self._validate_core_id(core_id)
        valid_temp = self._validate_temperature(temperature)
        valid_ts = self._validate_and_parse_timestamp(timestamp)

        if valid_core_id not in self._history_buffers:
            self._history_buffers[valid_core_id] = deque(maxlen=self.history_size)

        buf = self._history_buffers[valid_core_id]

        if valid_core_id not in self._states:
            # First reading for this core
            state = CoreThermalState(
                core_id=valid_core_id,
                current_temperature=valid_temp,
                previous_temperature=None,
                temperature_difference=None,
                current_timestamp=valid_ts,
                previous_timestamp=None,
                time_difference=None,
                history=[],
            )
        else:
            # Subsequent reading for this core
            prev_state = self._states[valid_core_id]
            prev_temp = prev_state.current_temperature
            prev_ts = prev_state.current_timestamp

            temp_diff = valid_temp - prev_temp
            time_diff = self._calculate_time_difference(valid_ts, prev_ts)

            state = CoreThermalState(
                core_id=valid_core_id,
                current_temperature=valid_temp,
                previous_temperature=prev_temp,
                temperature_difference=temp_diff,
                current_timestamp=valid_ts,
                previous_timestamp=prev_ts,
                time_difference=time_diff,
                history=[],
            )

        # Store in history buffer (oldest entries roll off automatically when history_size is reached)
        buf.append((valid_ts, valid_temp))
        state.history = list(buf)

        self._states[valid_core_id] = state
        return state

    def get_current_temperature(self, core_id: Union[int, str]) -> Optional[float]:
        """Return the current temperature for a core, or None if unmonitored."""
        state = self._states.get(core_id)
        return state.current_temperature if state is not None else None

    def get_previous_temperature(self, core_id: Union[int, str]) -> Optional[float]:
        """Return the previous temperature for a core, or None if unmonitored or first reading."""
        state = self._states.get(core_id)
        return state.previous_temperature if state is not None else None

    def get_temperature_difference(self, core_id: Union[int, str]) -> Optional[float]:
        """Return delta T (current - previous) for a core, or None if unmonitored or first reading."""
        state = self._states.get(core_id)
        return state.temperature_difference if state is not None else None

    def get_time_difference(self, core_id: Union[int, str]) -> Optional[float]:
        """Return delta t in seconds between current and previous reading, or None."""
        state = self._states.get(core_id)
        return state.time_difference if state is not None else None

    def get_temperature_history(self, core_id: Union[int, str]) -> List[Tuple[Any, float]]:
        """Return a copy of the stored (timestamp, temperature) history list for a core."""
        buf = self._history_buffers.get(core_id)
        if not buf:
            return []
        return list(buf)

    def get_latest_reading(self, core_id: Union[int, str]) -> Optional[CoreThermalState]:
        """Return a copy of the latest CoreThermalState for a core, or None if unmonitored."""
        state = self._states.get(core_id)
        if state is None:
            return None
        return CoreThermalState(
            core_id=state.core_id,
            current_temperature=state.current_temperature,
            previous_temperature=state.previous_temperature,
            temperature_difference=state.temperature_difference,
            current_timestamp=state.current_timestamp,
            previous_timestamp=state.previous_timestamp,
            time_difference=state.time_difference,
            history=list(state.history),
        )

    def get_all_core_temperatures(self) -> Dict[Union[int, str], float]:
        """Return dictionary mapping core_id to current_temperature for all active cores."""
        return {core_id: state.current_temperature for core_id, state in self._states.items()}

    def get_monitored_cores(self) -> List[Union[int, str]]:
        """Return list of currently monitored core IDs."""
        return list(self._states.keys())

    def reset(self, core_id: Optional[Union[int, str]] = None) -> None:
        """Reset internal state and history for a specific core, or all cores if core_id is None."""
        if core_id is None:
            self._states.clear()
            self._history_buffers.clear()
        else:
            self._states.pop(core_id, None)
            self._history_buffers.pop(core_id, None)

    # --------------------------------------------------------------------------
    # Private Input Validators & Helpers
    # --------------------------------------------------------------------------

    def _validate_core_id(self, core_id: Any) -> Union[int, str]:
        """Validate CPU core identifier.

        Raises:
            InvalidReadingError: If core_id is None, empty string, negative int, or invalid type.
        """
        if core_id is None:
            raise InvalidReadingError("Core ID cannot be None.")

        if isinstance(core_id, bool):
            raise InvalidReadingError(f"Core ID cannot be boolean, got {core_id}")

        if isinstance(core_id, int):
            if core_id < 0:
                raise InvalidReadingError(f"Integer core ID must be non-negative, got {core_id}")
            return core_id

        if isinstance(core_id, str):
            clean_id = core_id.strip()
            if not clean_id:
                raise InvalidReadingError("Core ID string cannot be empty or whitespace.")
            return clean_id

        raise InvalidReadingError(f"Core ID must be an int or str, got {type(core_id).__name__}: {core_id}")

    def _validate_temperature(self, temperature: Any) -> float:
        """Validate temperature value type, finiteness, and physical bounds.

        Raises:
            InvalidReadingError: If temperature is None, boolean, string, non-numeric, NaN, infinite, or out of bounds.
        """
        if temperature is None:
            raise InvalidReadingError("Temperature value cannot be None.")

        if isinstance(temperature, (bool, str)):
            raise InvalidReadingError(f"Temperature value cannot be {type(temperature).__name__}: {temperature}")

        if not isinstance(temperature, (int, float)):
            raise InvalidReadingError(f"Temperature must be numeric (int or float), got {type(temperature).__name__}: {temperature}")

        float_val = float(temperature)

        if not math.isfinite(float_val):
            raise InvalidReadingError(f"Temperature must be a finite number, got {float_val}.")

        if float_val < self.min_valid_temp or float_val > self.max_valid_temp:
            raise InvalidReadingError(
                f"Temperature {float_val} °C is outside valid physical bounds "
                f"[{self.min_valid_temp} °C, {self.max_valid_temp} °C]"
            )

        return float_val

    def _validate_and_parse_timestamp(
        self, timestamp: Optional[Union[float, int, datetime]]
    ) -> Union[float, int, datetime]:
        """Validate timestamp format."""
        if timestamp is None:
            return 0.0

        if isinstance(timestamp, bool):
            raise InvalidReadingError(f"Timestamp cannot be boolean, got {timestamp}")

        if isinstance(timestamp, (int, float)):
            float_ts = float(timestamp)
            if not math.isfinite(float_ts) or float_ts < 0.0:
                raise InvalidReadingError(f"Numeric timestamp must be non-negative and finite, got {timestamp}")
            return float_ts if isinstance(timestamp, float) else timestamp

        if isinstance(timestamp, datetime):
            return timestamp

        raise InvalidReadingError(f"Unsupported timestamp type {type(timestamp).__name__}: {timestamp}")

    def _calculate_time_difference(
        self, current_ts: Union[float, int, datetime], previous_ts: Union[float, int, datetime]
    ) -> Optional[float]:
        """Compute delta t in seconds between current and previous timestamp."""
        if type(current_ts) != type(previous_ts):
            return None

        if isinstance(current_ts, (int, float)) and isinstance(previous_ts, (int, float)):
            return float(current_ts - previous_ts)

        if isinstance(current_ts, datetime) and isinstance(previous_ts, datetime):
            return float((current_ts - previous_ts).total_seconds())

        return None
