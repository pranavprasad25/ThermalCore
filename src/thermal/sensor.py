"""Abstract thermal sensor interface and simulated thermal sensor implementation."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import random
from typing import Optional
from src.thermal.exceptions import SensorReadError
from src.thermal.reading import (
    ThermalReading,
    normalize_to_celsius,
    validate_and_normalize_unit,
    validate_and_parse_timestamp,
    validate_sensor_id,
)


class ThermalSensor(ABC):
    """Abstract Base Class representing a thermal sensor source."""

    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Return unique sensor identifier."""
        pass

    @abstractmethod
    def read(self) -> ThermalReading:
        """Acquire a temperature reading from the sensor source.

        Returns:
            ThermalReading: Standardized thermal reading object.

        Raises:
            SensorReadError: If sensor fails to acquire a reading.
        """
        pass


class SimulatedThermalSensor(ThermalSensor):
    """Simulated thermal sensor for development, testing, and workload emulation."""

    def __init__(
        self,
        sensor_id: str = "simulated_sensor_01",
        start_temp: float = 25.0,
        min_temp: float = -10.0,
        max_temp: float = 120.0,
        noise_std: float = 0.5,
        sampling_interval: float = 0.5,
        unit: str = "C",
        seed: Optional[int] = None,
    ):
        self._sensor_id = validate_sensor_id(sensor_id)
        self.current_temp = float(start_temp)
        self.target_temp = float(start_temp)
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        self.noise_std = float(noise_std)
        self.sampling_interval = float(sampling_interval)
        self.unit = validate_and_normalize_unit(unit)

        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._read_count = 0

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    def set_target_temperature(self, target_temp: float) -> None:
        """Set target temperature for simulated thermal drift/trend."""
        self.target_temp = float(target_temp)

    def read(self) -> ThermalReading:
        """Generate and return a simulated thermal reading."""
        # 1. Thermal drift toward target
        drift_rate = 0.1 * (self.target_temp - self.current_temp)
        self.current_temp += drift_rate

        # 2. Add Gaussian noise if noise_std > 0
        noise = self._rng.gauss(0, self.noise_std) if self.noise_std > 0 else 0.0
        raw_val = self.current_temp + noise

        # 3. Clamp between min_temp and max_temp
        clamped_raw = max(self.min_temp, min(self.max_temp, raw_val))

        # 4. Normalize to Celsius
        celsius_val = normalize_to_celsius(clamped_raw, self.unit)
        now = datetime.now(timezone.utc)
        self._read_count += 1

        return ThermalReading(
            temperature=celsius_val,
            timestamp=now,
            sensor_id=self._sensor_id,
            unit="C",
            raw_temperature=clamped_raw,
            original_unit=self.unit,
        )
