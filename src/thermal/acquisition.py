"""Thermal data acquisition pipeline, validation coordinator, and reading buffer."""

from collections import deque
from datetime import datetime
import time
from typing import Any, Generator, List, Optional, Union
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import (
    InvalidReadingError,
    SensorReadError,
    ThermalAcquisitionError,
)
from src.thermal.reading import (
    ThermalReading,
    normalize_to_celsius,
    validate_and_normalize_unit,
    validate_and_parse_timestamp,
    validate_sensor_id,
    validate_temperature_value,
)
from src.thermal.sensor import ThermalSensor


class ThermalReadingBuffer:
    """Lightweight rolling ring buffer storing recent thermal readings."""

    def __init__(self, max_capacity: int = 100):
        if max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {max_capacity}")
        self.max_capacity = max_capacity
        self._buffer: deque[ThermalReading] = deque(maxlen=max_capacity)

    def add(self, reading: ThermalReading) -> None:
        """Add a reading to the buffer. Rolls out oldest reading when full."""
        if not isinstance(reading, ThermalReading):
            raise InvalidReadingError(f"Buffer only accepts ThermalReading objects, got {type(reading).__name__}")
        self._buffer.append(reading)

    def get_recent(self, count: Optional[int] = None) -> List[ThermalReading]:
        """Retrieve the most recent N readings (ordered from oldest to newest)."""
        if count is None or count >= len(self._buffer):
            return list(self._buffer)
        if count <= 0:
            return []
        return list(self._buffer)[-count:]

    def get_all(self) -> List[ThermalReading]:
        """Retrieve all readings currently in buffer."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all stored readings in buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class ThermalDataAcquisition:
    """Coordinator for ingesting, validating, converting, and buffering thermal sensor data."""

    def __init__(
        self,
        config: Optional[ThermalConfig] = None,
        buffer_capacity: int = 100,
        min_valid_temp: float = -50.0,
        max_valid_temp: float = 150.0,
    ):
        self.config = config
        self.min_valid_temp = float(min_valid_temp)
        self.max_valid_temp = float(max_valid_temp)
        self.buffer = ThermalReadingBuffer(max_capacity=buffer_capacity)

    def ingest_reading(
        self,
        temperature: Any,
        sensor_id: str = "thermal_sensor_01",
        unit: str = "C",
        timestamp: Optional[Union[datetime, str, int, float]] = None,
    ) -> ThermalReading:
        """Ingest a raw temperature reading, validate, normalize to Celsius, and buffer it.

        Args:
            temperature: Raw temperature numeric input.
            sensor_id: Sensor identifier string.
            unit: Input temperature unit ("C", "F", "K").
            timestamp: Optional datetime or timestamp representation.

        Returns:
            ThermalReading: Validated and normalized thermal reading object.

        Raises:
            InvalidReadingError: If temperature is non-numeric, NaN, infinite, out of bounds, or missing.
            UnsupportedUnitError: If unit is unsupported.
        """
        # 1. Validate temperature numeric value
        raw_val = validate_temperature_value(temperature)

        # 2. Validate sensor ID
        clean_sensor_id = validate_sensor_id(sensor_id)

        # 3. Validate unit specification
        clean_unit = validate_and_normalize_unit(unit)

        # 4. Normalize temperature to Celsius
        celsius_val = normalize_to_celsius(raw_val, clean_unit)

        # 5. Operational temperature bounds check
        if not (self.min_valid_temp <= celsius_val <= self.max_valid_temp):
            raise InvalidReadingError(
                f"Temperature reading {celsius_val:.2f}°C out of valid physical bounds "
                f"[{self.min_valid_temp}°C, {self.max_valid_temp}°C]."
            )

        # 6. Validate and parse timestamp
        utc_timestamp = validate_and_parse_timestamp(timestamp)

        # 7. Construct standardized ThermalReading object
        reading = ThermalReading(
            temperature=celsius_val,
            timestamp=utc_timestamp,
            sensor_id=clean_sensor_id,
            unit="C",
            raw_temperature=raw_val,
            original_unit=clean_unit,
        )

        # 8. Buffer reading
        self.buffer.add(reading)
        return reading

    def read_from_sensor(self, sensor: ThermalSensor) -> Optional[ThermalReading]:
        """Safely acquire a reading from a ThermalSensor object.

        Handles sensor exceptions or missing data gracefully without crashing.

        Returns:
            ThermalReading: Successfully ingested reading, or None if sensor failed.
        """
        if sensor is None:
            return None

        try:
            raw_reading = sensor.read()
            if raw_reading is None:
                return None

            # Re-validate reading through ingestion pipeline
            return self.ingest_reading(
                temperature=raw_reading.temperature,
                sensor_id=raw_reading.sensor_id,
                unit=raw_reading.unit,
                timestamp=raw_reading.timestamp,
            )
        except ThermalAcquisitionError:
            # Re-raise acquisition domain errors if caller expects validation failure
            raise
        except Exception as err:
            # Treat underlying hardware/connection failures as SensorReadError
            raise SensorReadError(f"Sensor '{getattr(sensor, 'sensor_id', 'unknown')}' failed to read: {err}") from err

    def stream(
        self,
        sensor: ThermalSensor,
        count: Optional[int] = None,
        interval: float = 0.0,
    ) -> Generator[ThermalReading, None, None]:
        """Continuously stream thermal readings from a sensor source.

        Args:
            sensor: ThermalSensor source instance.
            count: Optional maximum number of readings to yield (prevents infinite loops).
            interval: Pause duration between readings in seconds.

        Yields:
            ThermalReading: Standardized thermal readings as acquired.
        """
        yielded = 0
        while True:
            if count is not None and yielded >= count:
                break

            try:
                reading = self.read_from_sensor(sensor)
                if reading is not None:
                    yielded += 1
                    yield reading
            except SensorReadError:
                pass  # Gracefully skip sensor failure in stream

            if count is not None and yielded >= count:
                break

            if interval > 0:
                time.sleep(interval)
