"""Thermal data processing engine, statistical calculations, and trend analysis."""

from typing import Dict, List, Optional
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidReadingError
from src.thermal.history import ThermalHistoryBuffer
from src.thermal.processed_data import ProcessedThermalData, ThermalTrend
from src.thermal.reading import ThermalReading


class ThermalDataProcessor:
    """Processor ingesting raw ThermalReadings, performing EMA smoothing, statistics, and trend detection."""

    def __init__(self, config: Optional[ThermalConfig] = None):
        self.config = config if config is not None else ThermalConfig()
        self.history = ThermalHistoryBuffer(max_capacity=self.config.history_capacity)
        # Tracking Exponential Moving Average (EMA) per sensor: {sensor_id: smoothed_val}
        self._smoothed_temps: Dict[str, float] = {}

    def process(self, reading: ThermalReading) -> ProcessedThermalData:
        """Process an incoming ThermalReading, apply smoothing, statistics, rate of change, and trend classification."""
        if not isinstance(reading, ThermalReading):
            raise InvalidReadingError(f"Processor accepts ThermalReading instances, got {type(reading).__name__}")

        sensor_id = reading.sensor_id
        raw_temp = reading.temperature

        # 1. Add reading to multi-sensor history
        self.history.add(reading)

        # 2. Noise Filtering / Smoothing (Exponential Moving Average)
        if self.config.smoothing_enabled:
            alpha = self.config.smoothing_alpha
            if sensor_id not in self._smoothed_temps:
                smoothed_temp = raw_temp
            else:
                prev_smoothed = self._smoothed_temps[sensor_id]
                smoothed_temp = (alpha * raw_temp) + ((1.0 - alpha) * prev_smoothed)
            self._smoothed_temps[sensor_id] = float(smoothed_temp)
        else:
            smoothed_temp = raw_temp

        # 3. Calculate Moving Average over configured window
        moving_avg = self.get_moving_average(sensor_id, window=self.config.moving_average_window)

        # 4. Calculate Rate of Change (°C/sec)
        rate_of_change = self.get_rate_of_change(sensor_id, unit_time="second")

        # 5. Determine Heating / Cooling / Stable Trend
        trend = self.get_trend(sensor_id)

        # 6. Construct ProcessedThermalData
        return ProcessedThermalData(
            raw_reading=reading,
            processed_temperature=float(smoothed_temp),
            moving_average=float(moving_avg),
            rate_of_change=float(rate_of_change),
            trend=trend,
        )

    def get_current_temperature(self, sensor_id: str) -> float:
        """Return the current processed/smoothed temperature for sensor_id."""
        latest_reading = self.history.latest(sensor_id)
        if latest_reading is None:
            raise InvalidReadingError(f"No thermal history available for sensor '{sensor_id}'")

        if self.config.smoothing_enabled and sensor_id in self._smoothed_temps:
            return self._smoothed_temps[sensor_id]
        return latest_reading.temperature

    def get_min_temperature(self, sensor_id: str, window: Optional[int] = None) -> float:
        """Calculate minimum temperature over history or recent window for sensor_id."""
        readings = self.history.recent(sensor_id, window) if window else self.history.all(sensor_id)
        if not readings:
            raise InvalidReadingError(f"No thermal history available for sensor '{sensor_id}'")
        return min(r.temperature for r in readings)

    def get_max_temperature(self, sensor_id: str, window: Optional[int] = None) -> float:
        """Calculate maximum temperature over history or recent window for sensor_id."""
        readings = self.history.recent(sensor_id, window) if window else self.history.all(sensor_id)
        if not readings:
            raise InvalidReadingError(f"No thermal history available for sensor '{sensor_id}'")
        return max(r.temperature for r in readings)

    def get_average_temperature(self, sensor_id: str, window: Optional[int] = None) -> float:
        """Calculate average temperature over history or recent window for sensor_id."""
        readings = self.history.recent(sensor_id, window) if window else self.history.all(sensor_id)
        if not readings:
            raise InvalidReadingError(f"No thermal history available for sensor '{sensor_id}'")
        return sum(r.temperature for r in readings) / len(readings)

    def get_moving_average(self, sensor_id: str, window: Optional[int] = None) -> float:
        """Calculate moving average temperature over window for sensor_id."""
        w_size = window if window is not None else self.config.moving_average_window
        if w_size <= 0:
            raise InvalidReadingError(f"Moving average window must be positive, got {w_size}")

        readings = self.history.recent(sensor_id, w_size)
        if not readings:
            raise InvalidReadingError(f"No thermal history available for sensor '{sensor_id}'")
        return sum(r.temperature for r in readings) / len(readings)

    def get_rate_of_change(self, sensor_id: str, unit_time: str = "second") -> float:
        """Calculate temperature change rate (ΔT / Δt) for sensor_id.

        Args:
            sensor_id: Sensor identifier string.
            unit_time: Time unit for rate ("second" or "minute").

        Returns:
            float: Rate of change in °C/sec or °C/min.
        """
        readings = self.history.recent(sensor_id, 2)
        if len(readings) < 2:
            return 0.0

        r_prev, r_latest = readings[0], readings[1]
        delta_temp = r_latest.temperature - r_prev.temperature
        delta_time = (r_latest.timestamp - r_prev.timestamp).total_seconds()

        if delta_time <= 0:
            return 0.0  # Handle zero or negative timestamp difference safely

        rate_per_sec = delta_temp / delta_time
        if unit_time.lower() == "minute":
            return rate_per_sec * 60.0
        return rate_per_sec

    def get_trend(self, sensor_id: str) -> ThermalTrend:
        """Identify thermal trend (HEATING, COOLING, STABLE, UNKNOWN) for sensor_id."""
        readings = self.history.recent(sensor_id, 2)
        if len(readings) < 2:
            return ThermalTrend.UNKNOWN

        rate = self.get_rate_of_change(sensor_id, unit_time="second")
        tolerance = self.config.trend_tolerance

        if rate > tolerance:
            return ThermalTrend.HEATING
        elif rate < -tolerance:
            return ThermalTrend.COOLING
        else:
            return ThermalTrend.STABLE

    def get_history(self, sensor_id: str) -> List[ThermalReading]:
        """Retrieve full stored raw history for sensor_id."""
        return self.history.all(sensor_id)

    def get_recent_history(self, sensor_id: str, n: int = 10) -> List[ThermalReading]:
        """Retrieve N recent raw readings for sensor_id."""
        return self.history.recent(sensor_id, n)

    def clear_history(self, sensor_id: Optional[str] = None) -> None:
        """Clear history and smoothed state for a specific sensor_id, or all sensors."""
        self.history.clear(sensor_id)
        if sensor_id is None:
            self._smoothed_temps.clear()
        elif sensor_id in self._smoothed_temps:
            del self._smoothed_temps[sensor_id]
