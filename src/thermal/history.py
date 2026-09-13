"""Multi-sensor chronological thermal history buffer."""

from collections import deque
from typing import Dict, List, Optional
from src.thermal.exceptions import InvalidReadingError
from src.thermal.reading import ThermalReading


class ThermalHistoryBuffer:
    """Bounded per-sensor thermal reading history buffer maintaining strict chronological ordering."""

    def __init__(self, max_capacity: int = 100):
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {max_capacity}")
        self.max_capacity = max_capacity
        self._buffers: Dict[str, deque[ThermalReading]] = {}

    def add(self, reading: ThermalReading) -> None:
        """Add a reading to the buffer for its associated sensor_id.

        Maintains chronological ordering and rolls out oldest entries when max_capacity is reached.
        """
        if not isinstance(reading, ThermalReading):
            raise InvalidReadingError(f"Buffer accepts ThermalReading instances, got {type(reading).__name__}")

        sensor_id = reading.sensor_id
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self.max_capacity)

        buf = self._buffers[sensor_id]

        # Handle out-of-order timestamp insertion gracefully
        if buf and reading.timestamp < buf[-1].timestamp:
            # Sort full buffer chronologically if out-of-order timestamp received
            combined = list(buf) + [reading]
            combined.sort(key=lambda r: r.timestamp)
            self._buffers[sensor_id] = deque(combined[-self.max_capacity:], maxlen=self.max_capacity)
        else:
            buf.append(reading)

    def latest(self, sensor_id: str) -> Optional[ThermalReading]:
        """Return the most recent ThermalReading for a given sensor_id, or None if empty."""
        buf = self._buffers.get(sensor_id)
        if not buf:
            return None
        return buf[-1]

    def recent(self, sensor_id: str, n: int = 10) -> List[ThermalReading]:
        """Retrieve up to N recent readings for sensor_id in chronological order."""
        buf = self._buffers.get(sensor_id)
        if not buf:
            return []
        if n <= 0:
            return []
        if n >= len(buf):
            return list(buf)
        return list(buf)[-n:]

    def all(self, sensor_id: str) -> List[ThermalReading]:
        """Retrieve all stored readings for a given sensor_id."""
        buf = self._buffers.get(sensor_id)
        if not buf:
            return []
        return list(buf)

    def clear(self, sensor_id: Optional[str] = None) -> None:
        """Clear history for a specific sensor_id, or all sensors if sensor_id is None."""
        if sensor_id is None:
            self._buffers.clear()
        elif sensor_id in self._buffers:
            self._buffers[sensor_id].clear()

    def size(self, sensor_id: Optional[str] = None) -> int:
        """Return total number of stored readings for sensor_id, or total across all sensors if None."""
        if sensor_id is None:
            return sum(len(b) for b in self._buffers.values())
        return len(self._buffers.get(sensor_id, deque()))

    def get_sensor_ids(self) -> List[str]:
        """Return list of active sensor IDs in history."""
        return list(self._buffers.keys())
