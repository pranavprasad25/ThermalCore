"""Automated unit tests for Day 2 — Task 1: Thermal Data Acquisition."""

from datetime import datetime, timezone, timedelta
import math
import unittest
from src.thermal.acquisition import ThermalDataAcquisition, ThermalReadingBuffer
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import (
    InvalidReadingError,
    SensorReadError,
    ThermalAcquisitionError,
    UnsupportedUnitError,
)
from src.thermal.reading import ThermalReading, normalize_to_celsius
from src.thermal.sensor import SimulatedThermalSensor, ThermalSensor


class FailingMockSensor(ThermalSensor):
    """Mock sensor that intentionally raises an exception on read."""

    def __init__(self, sensor_id: str = "failing_sensor_01"):
        self._sensor_id = sensor_id

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    def read(self) -> ThermalReading:
        raise RuntimeError("Hardware connection lost.")


class NullMockSensor(ThermalSensor):
    """Mock sensor that returns None on read."""

    @property
    def sensor_id(self) -> str:
        return "null_sensor_01"

    def read(self) -> ThermalReading:
        return None  # type: ignore


class TestThermalDataAcquisition(unittest.TestCase):
    """Test suite covering the 20 mandatory test requirements for Day 2 Task 1."""

    def setUp(self) -> None:
        self.config = ThermalConfig(min_valid_temp=-50.0, max_valid_temp=150.0)
        self.acq = ThermalDataAcquisition(config=self.config, buffer_capacity=10)

    def test_1_valid_temperature_reading(self) -> None:
        """Test 1: Valid temperature reading ingestion."""
        reading = self.acq.ingest_reading(temperature=72.5, sensor_id="sensor_01", unit="C")
        self.assertIsInstance(reading, ThermalReading)
        self.assertEqual(reading.temperature, 72.5)
        self.assertEqual(reading.sensor_id, "sensor_01")
        self.assertEqual(reading.unit, "C")

    def test_2_automatic_timestamp_generation(self) -> None:
        """Test 2: Automatic timestamp generation when no timestamp provided."""
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        reading = self.acq.ingest_reading(temperature=45.0, sensor_id="sensor_01")
        after = datetime.now(timezone.utc) + timedelta(seconds=1)

        self.assertIsNotNone(reading.timestamp)
        self.assertIsNotNone(reading.timestamp.tzinfo)
        self.assertTrue(before <= reading.timestamp <= after)

    def test_3_provided_timestamp(self) -> None:
        """Test 3: Provided timestamp handling (datetime object and ISO string)."""
        custom_dt = datetime(2026, 9, 13, 10, 0, 0, tzinfo=timezone.utc)
        reading_dt = self.acq.ingest_reading(temperature=50.0, timestamp=custom_dt)
        self.assertEqual(reading_dt.timestamp, custom_dt)

        iso_str = "2026-09-13T10:30:00+00:00"
        reading_iso = self.acq.ingest_reading(temperature=55.0, timestamp=iso_str)
        self.assertEqual(reading_iso.timestamp.year, 2026)
        self.assertEqual(reading_iso.timestamp.hour, 10)

    def test_4_sensor_id_handling(self) -> None:
        """Test 4: Sensor ID preservation and normalization."""
        reading = self.acq.ingest_reading(temperature=30.0, sensor_id="  thermal_core_sensor_42  ")
        self.assertEqual(reading.sensor_id, "thermal_core_sensor_42")

    def test_5_unit_handling(self) -> None:
        """Test 5: Temperature unit conversion (°F and K converted to °C)."""
        # 98.6°F == 37.0°C
        reading_f = self.acq.ingest_reading(temperature=98.6, unit="F")
        self.assertAlmostEqual(reading_f.temperature, 37.0, places=1)
        self.assertEqual(reading_f.unit, "C")
        self.assertEqual(reading_f.original_unit, "F")

        # 300.15 K == 27.0°C
        reading_k = self.acq.ingest_reading(temperature=300.15, unit="K")
        self.assertAlmostEqual(reading_k.temperature, 27.0, places=1)

    def test_6_invalid_temperature_type(self) -> None:
        """Test 6: Rejection of invalid temperature types (non-numeric, string, dict, bool)."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature="invalid_str")

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=True)

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature={"temp": 25.0})

    def test_7_missing_temperature(self) -> None:
        """Test 7: Rejection of None temperature value."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=None)

    def test_8_nan_temperature(self) -> None:
        """Test 8: Rejection of NaN temperature values."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=float("nan"))

    def test_9_infinity_temperature(self) -> None:
        """Test 9: Rejection of positive and negative infinity values."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=float("inf"))

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=float("-inf"))

    def test_10_invalid_timestamp(self) -> None:
        """Test 10: Rejection of malformed timestamps."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=25.0, timestamp="not_a_timestamp")

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=25.0, timestamp=-100)

    def test_11_invalid_sensor_id(self) -> None:
        """Test 11: Rejection of empty or invalid sensor IDs."""
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=25.0, sensor_id="")

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=25.0, sensor_id="   ")

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=25.0, sensor_id=None)

    def test_12_missing_sensor_reading(self) -> None:
        """Test 12: Safe handling when sensor returns None."""
        sensor = NullMockSensor()
        result = self.acq.read_from_sensor(sensor)
        self.assertIsNone(result)

    def test_13_simulated_sensor_output(self) -> None:
        """Test 13: Verification of SimulatedThermalSensor output format."""
        sensor = SimulatedThermalSensor(sensor_id="sim_01", start_temp=40.0, noise_std=0.0, seed=42)
        reading = sensor.read()
        self.assertIsInstance(reading, ThermalReading)
        self.assertEqual(reading.sensor_id, "sim_01")
        self.assertAlmostEqual(reading.temperature, 40.0, places=1)

    def test_14_multiple_sequential_readings(self) -> None:
        """Test 14: Ingestion and tracking of multiple sequential readings."""
        sensor = SimulatedThermalSensor(sensor_id="sim_01", start_temp=30.0, noise_std=0.1, seed=123)
        readings = [self.acq.read_from_sensor(sensor) for _ in range(5)]

        self.assertEqual(len(readings), 5)
        for r in readings:
            self.assertIsNotNone(r)
            self.assertIsInstance(r, ThermalReading)

        buffer_items = self.acq.buffer.get_all()
        self.assertEqual(len(buffer_items), 5)

    def test_15_continuous_stream_with_finite_count(self) -> None:
        """Test 15: Stream generation with finite sample count limit."""
        sensor = SimulatedThermalSensor(sensor_id="sim_stream", start_temp=25.0, seed=7)
        streamed = list(self.acq.stream(sensor, count=5, interval=0.0))

        self.assertEqual(len(streamed), 5)
        for r in streamed:
            self.assertEqual(r.sensor_id, "sim_stream")

    def test_16_configurable_sampling_behavior(self) -> None:
        """Test 16: Configurable sampling parameters on SimulatedThermalSensor."""
        sensor = SimulatedThermalSensor(
            sensor_id="custom_sim",
            start_temp=50.0,
            min_temp=20.0,
            max_temp=80.0,
            unit="F",  # 50°F == 10°C, clamped to min 20°F == -6.67°C
            seed=1,
        )
        reading = sensor.read()
        self.assertEqual(reading.original_unit, "F")
        self.assertEqual(reading.unit, "C")

    def test_17_buffer_history_behavior(self) -> None:
        """Test 17: Buffer rollout when max capacity is reached."""
        buf = ThermalReadingBuffer(max_capacity=3)
        readings = [
            ThermalReading(temperature=float(i), timestamp=datetime.now(timezone.utc), sensor_id="s1")
            for i in range(5)
        ]

        for r in readings:
            buf.add(r)

        self.assertEqual(len(buf), 3)
        recent = buf.get_all()
        self.assertEqual([r.temperature for r in recent], [2.0, 3.0, 4.0])

        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_18_sensor_failure_handling(self) -> None:
        """Test 18: Graceful handling of sensor read failure exceptions."""
        failing_sensor = FailingMockSensor()
        with self.assertRaises(SensorReadError):
            self.acq.read_from_sensor(failing_sensor)

    def test_19_boundary_temperature_values(self) -> None:
        """Test 19: Handling of valid boundary temperature values and physical range check."""
        r_min = self.acq.ingest_reading(temperature=-50.0)
        self.assertEqual(r_min.temperature, -50.0)

        r_max = self.acq.ingest_reading(temperature=150.0)
        self.assertEqual(r_max.temperature, 150.0)

        # Exceeding physical bounds raises InvalidReadingError
        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=-50.1)

        with self.assertRaises(InvalidReadingError):
            self.acq.ingest_reading(temperature=150.1)

    def test_20_end_to_end_acquisition_flow(self) -> None:
        """Test 20: Complete end-to-end ingestion, conversion, serialization, and buffering pipeline."""
        sensor = SimulatedThermalSensor(sensor_id="e2e_sensor", start_temp=60.0, seed=99)
        reading = self.acq.read_from_sensor(sensor)

        self.assertIsNotNone(reading)
        data_dict = reading.to_dict()

        self.assertIn("temperature", data_dict)
        self.assertIn("timestamp", data_dict)
        self.assertEqual(data_dict["sensor_id"], "e2e_sensor")
        self.assertEqual(data_dict["unit"], "C")

        # Verify buffer contains the exact reading
        buffered = self.acq.buffer.get_recent(1)[0]
        self.assertEqual(buffered.temperature, reading.temperature)


if __name__ == "__main__":
    unittest.main()
