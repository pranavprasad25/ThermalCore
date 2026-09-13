"""Automated unit tests for Day 2 — Task 2: Thermal Data Processing & History."""

from datetime import datetime, timezone, timedelta
import unittest
from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidReadingError, SensorReadError
from src.thermal.history import ThermalHistoryBuffer
from src.thermal.processed_data import ProcessedThermalData, ThermalTrend
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading
from src.thermal.sensor import SimulatedThermalSensor


class TestThermalDataProcessor(unittest.TestCase):
    """Test suite covering the 44 mandatory test requirements for Day 2 Task 2."""

    def setUp(self) -> None:
        self.config = ThermalConfig(
            history_capacity=10,
            moving_average_window=3,
            smoothing_enabled=True,
            smoothing_alpha=0.5,
            trend_tolerance=0.05,
        )
        self.processor = ThermalDataProcessor(config=self.config)
        self.base_time = datetime(2026, 9, 13, 10, 0, 0, tzinfo=timezone.utc)

    def _create_reading(self, temp: float, seconds_offset: float = 0.0, sensor_id: str = "sensor_01") -> ThermalReading:
        ts = self.base_time + timedelta(seconds=seconds_offset)
        return ThermalReading(temperature=temp, timestamp=ts, sensor_id=sensor_id)

    # --- History Tests (1-10) ---

    def test_01_adding_a_reading(self) -> None:
        """Test 1: Adding a reading to history."""
        r = self._create_reading(50.0)
        self.processor.process(r)
        self.assertEqual(self.processor.history.size("sensor_01"), 1)

    def test_02_retrieving_latest_reading(self) -> None:
        """Test 2: Retrieving latest reading."""
        self.processor.process(self._create_reading(50.0, 0))
        self.processor.process(self._create_reading(55.0, 1))
        latest = self.processor.history.latest("sensor_01")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.temperature, 55.0)

    def test_03_retrieving_recent_readings(self) -> None:
        """Test 3: Retrieving N recent readings."""
        for i in range(5):
            self.processor.process(self._create_reading(50.0 + i, i))
        recent = self.processor.get_recent_history("sensor_01", n=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual([r.temperature for r in recent], [52.0, 53.0, 54.0])

    def test_04_retrieving_full_history(self) -> None:
        """Test 4: Retrieving full history for a sensor."""
        for i in range(4):
            self.processor.process(self._create_reading(40.0 + i, i))
        full = self.processor.get_history("sensor_01")
        self.assertEqual(len(full), 4)

    def test_05_clearing_history(self) -> None:
        """Test 5: Clearing history."""
        self.processor.process(self._create_reading(50.0))
        self.processor.clear_history("sensor_01")
        self.assertEqual(self.processor.history.size("sensor_01"), 0)

    def test_06_history_size(self) -> None:
        """Test 6: Checking history size."""
        self.assertEqual(self.processor.history.size("sensor_01"), 0)
        self.processor.process(self._create_reading(50.0))
        self.assertEqual(self.processor.history.size("sensor_01"), 1)

    def test_07_maximum_history_limit(self) -> None:
        """Test 7: Maximum history capacity limit enforcement."""
        for i in range(15):
            self.processor.process(self._create_reading(30.0 + i, i))
        self.assertEqual(self.processor.history.size("sensor_01"), 10)

    def test_08_oldest_reading_removal(self) -> None:
        """Test 8: Oldest reading rollout when buffer capacity is reached."""
        for i in range(12):
            self.processor.process(self._create_reading(10.0 + i, i))
        history = self.processor.get_history("sensor_01")
        self.assertEqual(history[0].temperature, 12.0)  # 10.0 and 11.0 rolled out
        self.assertEqual(history[-1].temperature, 21.0)

    def test_09_chronological_ordering(self) -> None:
        """Test 9: Chronological ordering maintenance even with out-of-order timestamps."""
        r1 = self._create_reading(50.0, seconds_offset=10)
        r2 = self._create_reading(40.0, seconds_offset=5)
        self.processor.history.add(r1)
        self.processor.history.add(r2)

        ordered = self.processor.get_history("sensor_01")
        self.assertEqual(ordered[0].temperature, 40.0)
        self.assertEqual(ordered[1].temperature, 50.0)

    def test_10_empty_history(self) -> None:
        """Test 10: Behavior on empty history."""
        self.assertIsNone(self.processor.history.latest("sensor_01"))
        self.assertEqual(self.processor.get_history("sensor_01"), [])

    # --- Statistics Tests (11-17) ---

    def test_11_current_temperature(self) -> None:
        """Test 11: Current temperature retrieval."""
        self.processor.process(self._create_reading(60.0))
        self.assertEqual(self.processor.get_current_temperature("sensor_01"), 60.0)

    def test_12_minimum_temperature(self) -> None:
        """Test 12: Minimum temperature calculation."""
        for t in [70.0, 65.0, 80.0, 62.0]:
            self.processor.process(self._create_reading(t))
        self.assertEqual(self.processor.get_min_temperature("sensor_01"), 62.0)

    def test_13_maximum_temperature(self) -> None:
        """Test 13: Maximum temperature calculation."""
        for t in [70.0, 65.0, 80.0, 62.0]:
            self.processor.process(self._create_reading(t))
        self.assertEqual(self.processor.get_max_temperature("sensor_01"), 80.0)

    def test_14_average_temperature(self) -> None:
        """Test 14: Average temperature calculation."""
        for t in [60.0, 70.0, 80.0]:
            self.processor.process(self._create_reading(t))
        self.assertEqual(self.processor.get_average_temperature("sensor_01"), 70.0)

    def test_15_single_reading_statistics(self) -> None:
        """Test 15: Statistics calculations on single reading."""
        self.processor.process(self._create_reading(55.0))
        self.assertEqual(self.processor.get_min_temperature("sensor_01"), 55.0)
        self.assertEqual(self.processor.get_max_temperature("sensor_01"), 55.0)
        self.assertEqual(self.processor.get_average_temperature("sensor_01"), 55.0)

    def test_16_multiple_reading_statistics(self) -> None:
        """Test 16: Statistics calculations on recent window subset."""
        for t in [100.0, 50.0, 60.0, 70.0]:
            self.processor.process(self._create_reading(t))
        # Window of last 3 readings: [50, 60, 70]
        self.assertEqual(self.processor.get_min_temperature("sensor_01", window=3), 50.0)
        self.assertEqual(self.processor.get_max_temperature("sensor_01", window=3), 70.0)
        self.assertEqual(self.processor.get_average_temperature("sensor_01", window=3), 60.0)

    def test_17_empty_history_behavior(self) -> None:
        """Test 17: Empty history exception handling."""
        with self.assertRaises(InvalidReadingError):
            self.processor.get_current_temperature("non_existent_sensor")
        with self.assertRaises(InvalidReadingError):
            self.processor.get_min_temperature("non_existent_sensor")
        with self.assertRaises(InvalidReadingError):
            self.processor.get_max_temperature("non_existent_sensor")
        with self.assertRaises(InvalidReadingError):
            self.processor.get_average_temperature("non_existent_sensor")

    # --- Moving Average Tests (18-22) ---

    def test_18_correct_moving_average(self) -> None:
        """Test 18: Moving average calculation over window."""
        for t in [70.0, 72.0, 74.0, 76.0, 78.0]:
            self.processor.process(self._create_reading(t))
        # Last 3 readings: 74, 76, 78 -> avg = 76.0
        self.assertEqual(self.processor.get_moving_average("sensor_01", window=3), 76.0)

    def test_19_different_window_sizes(self) -> None:
        """Test 19: Moving average with different window sizes."""
        for t in [10.0, 20.0, 30.0, 40.0]:
            self.processor.process(self._create_reading(t))
        self.assertEqual(self.processor.get_moving_average("sensor_01", window=2), 35.0)
        self.assertEqual(self.processor.get_moving_average("sensor_01", window=4), 25.0)

    def test_20_window_larger_than_history(self) -> None:
        """Test 20: Moving average when window size exceeds available history."""
        self.processor.process(self._create_reading(50.0))
        self.processor.process(self._create_reading(60.0))
        # History size = 2, requested window = 10 -> averages available 2 readings (55.0)
        self.assertEqual(self.processor.get_moving_average("sensor_01", window=10), 55.0)

    def test_21_invalid_window_size(self) -> None:
        """Test 21: Rejection of non-positive window sizes."""
        self.processor.process(self._create_reading(50.0))
        with self.assertRaises(InvalidReadingError):
            self.processor.get_moving_average("sensor_01", window=0)
        with self.assertRaises(InvalidReadingError):
            self.processor.get_moving_average("sensor_01", window=-5)

    def test_22_single_reading_window(self) -> None:
        """Test 22: Moving average with window size of 1."""
        for t in [50.0, 60.0, 75.0]:
            self.processor.process(self._create_reading(t))
        self.assertEqual(self.processor.get_moving_average("sensor_01", window=1), 75.0)

    # --- Rate of Change Tests (23-29) ---

    def test_23_positive_temperature_change(self) -> None:
        """Test 23: Positive temperature rate of change."""
        self.processor.process(self._create_reading(70.0, seconds_offset=0))
        self.processor.process(self._create_reading(75.0, seconds_offset=10))
        # ΔT = 5.0, Δt = 10s -> 0.5 °C/s
        self.assertAlmostEqual(self.processor.get_rate_of_change("sensor_01", unit_time="second"), 0.5)

    def test_24_negative_temperature_change(self) -> None:
        """Test 24: Negative temperature rate of change."""
        self.processor.process(self._create_reading(80.0, seconds_offset=0))
        self.processor.process(self._create_reading(70.0, seconds_offset=5))
        # ΔT = -10.0, Δt = 5s -> -2.0 °C/s
        self.assertAlmostEqual(self.processor.get_rate_of_change("sensor_01", unit_time="second"), -2.0)

    def test_25_zero_change(self) -> None:
        """Test 25: Zero temperature rate of change."""
        self.processor.process(self._create_reading(70.0, seconds_offset=0))
        self.processor.process(self._create_reading(70.0, seconds_offset=10))
        self.assertEqual(self.processor.get_rate_of_change("sensor_01"), 0.0)

    def test_26_correct_time_conversion(self) -> None:
        """Test 26: Rate of change time unit conversion (°C/sec vs °C/min)."""
        self.processor.process(self._create_reading(70.0, seconds_offset=0))
        self.processor.process(self._create_reading(75.0, seconds_offset=60))
        # ΔT = 5°C over 60s -> 5.0 °C/min or 0.0833 °C/sec
        self.assertAlmostEqual(self.processor.get_rate_of_change("sensor_01", unit_time="minute"), 5.0)

    def test_27_multiple_time_intervals(self) -> None:
        """Test 27: Rate of change calculated over latest timestamp delta."""
        self.processor.process(self._create_reading(60.0, seconds_offset=0))
        self.processor.process(self._create_reading(65.0, seconds_offset=10))
        self.processor.process(self._create_reading(75.0, seconds_offset=12))
        # Latest delta: 65 -> 75 (ΔT = 10) in 2 seconds -> +5.0 °C/s
        self.assertAlmostEqual(self.processor.get_rate_of_change("sensor_01"), 5.0)

    def test_28_zero_timestamp_difference(self) -> None:
        """Test 28: Zero timestamp difference handled safely without division by zero."""
        self.processor.process(self._create_reading(70.0, seconds_offset=0))
        self.processor.process(self._create_reading(75.0, seconds_offset=0))
        self.assertEqual(self.processor.get_rate_of_change("sensor_01"), 0.0)

    def test_29_insufficient_readings_rate(self) -> None:
        """Test 29: Rate of change with less than 2 readings returns 0.0."""
        self.processor.process(self._create_reading(70.0))
        self.assertEqual(self.processor.get_rate_of_change("sensor_01"), 0.0)

    # --- Trend Tests (30-34) ---

    def test_30_heating_trend(self) -> None:
        """Test 30: HEATING trend classification."""
        self.processor.process(self._create_reading(60.0, seconds_offset=0))
        self.processor.process(self._create_reading(65.0, seconds_offset=5))
        self.assertEqual(self.processor.get_trend("sensor_01"), ThermalTrend.HEATING)

    def test_31_cooling_trend(self) -> None:
        """Test 31: COOLING trend classification."""
        self.processor.process(self._create_reading(80.0, seconds_offset=0))
        self.processor.process(self._create_reading(75.0, seconds_offset=5))
        self.assertEqual(self.processor.get_trend("sensor_01"), ThermalTrend.COOLING)

    def test_32_stable_trend(self) -> None:
        """Test 32: STABLE trend classification."""
        self.processor.process(self._create_reading(70.0, seconds_offset=0))
        self.processor.process(self._create_reading(70.1, seconds_offset=10))
        # Rate = 0.01 °C/s <= tolerance (0.05) -> STABLE
        self.assertEqual(self.processor.get_trend("sensor_01"), ThermalTrend.STABLE)

    def test_33_insufficient_data_trend(self) -> None:
        """Test 33: UNKNOWN trend on insufficient data."""
        self.assertEqual(self.processor.get_trend("sensor_01"), ThermalTrend.UNKNOWN)
        self.processor.process(self._create_reading(70.0))
        self.assertEqual(self.processor.get_trend("sensor_01"), ThermalTrend.UNKNOWN)

    def test_34_stability_tolerance(self) -> None:
        """Test 34: Configurable stability tolerance check."""
        strict_config = ThermalConfig(trend_tolerance=0.001)
        strict_processor = ThermalDataProcessor(config=strict_config)
        strict_processor.process(self._create_reading(70.0, seconds_offset=0))
        strict_processor.process(self._create_reading(70.1, seconds_offset=10))
        # Rate = 0.01 °C/s > tolerance 0.001 -> HEATING
        self.assertEqual(strict_processor.get_trend("sensor_01"), ThermalTrend.HEATING)

    # --- Smoothing Tests (35-39) ---

    def test_35_smoothing_enabled(self) -> None:
        """Test 35: EMA smoothing behavior when enabled."""
        r1 = self.processor.process(self._create_reading(70.0, 0))
        self.assertEqual(r1.processed_temperature, 70.0)

        # alpha = 0.5: S_t = 0.5 * 80.0 + 0.5 * 70.0 = 75.0
        r2 = self.processor.process(self._create_reading(80.0, 1))
        self.assertEqual(r2.processed_temperature, 75.0)

    def test_36_smoothing_disabled(self) -> None:
        """Test 36: Behavior when smoothing is disabled."""
        disabled_config = ThermalConfig(smoothing_enabled=False)
        p = ThermalDataProcessor(config=disabled_config)
        p.process(self._create_reading(70.0, 0))
        res = p.process(self._create_reading(80.0, 1))
        self.assertEqual(res.processed_temperature, 80.0)

    def test_37_correct_smoothing_calculation(self) -> None:
        """Test 37: Mathematical accuracy of Exponential Moving Average smoothing."""
        p = ThermalDataProcessor(config=ThermalConfig(smoothing_alpha=0.2))
        p.process(self._create_reading(100.0, 0))  # S0 = 100.0
        res = p.process(self._create_reading(50.0, 1))  # S1 = 0.2*50 + 0.8*100 = 90.0
        self.assertAlmostEqual(res.processed_temperature, 90.0)

    def test_38_raw_data_remains_unchanged(self) -> None:
        """Test 38: Immutability of raw ThermalReading data after smoothing."""
        r1 = self._create_reading(70.0, 0)
        res1 = self.processor.process(r1)
        r2 = self._create_reading(80.0, 1)
        res2 = self.processor.process(r2)

        self.assertEqual(res2.raw_reading.temperature, 80.0)
        self.assertEqual(res2.processed_temperature, 75.0)
        self.assertEqual(r2.temperature, 80.0)

    def test_39_noise_reduction_behavior(self) -> None:
        """Test 39: Reduction of artificial noise in processed output compared to raw variance."""
        noisy_temps = [70.0, 74.0, 68.0, 75.0, 71.0]
        results = [self.processor.process(self._create_reading(t, i)) for i, t in enumerate(noisy_temps)]

        raws = [r.raw_reading.temperature for r in results]
        smooths = [r.processed_temperature for r in results]

        # Calculate variance
        raw_mean = sum(raws) / len(raws)
        raw_var = sum((x - raw_mean)**2 for x in raws) / len(raws)

        smooth_mean = sum(smooths) / len(smooths)
        smooth_var = sum((x - smooth_mean)**2 for x in smooths) / len(smooths)

        self.assertLess(smooth_var, raw_var)

    # --- Integration & Multi-Sensor Tests (40-44) ---

    def test_40_task1_to_task2_end_to_end_processing(self) -> None:
        """Test 40: End-to-end flow from Task 1 raw reading to Task 2 ProcessedThermalData."""
        r = ThermalReading(temperature=65.0, timestamp=self.base_time, sensor_id="sensor_01")
        processed = self.processor.process(r)

        self.assertIsInstance(processed, ProcessedThermalData)
        self.assertEqual(processed.raw_reading.temperature, 65.0)
        self.assertEqual(processed.processed_temperature, 65.0)

    def test_41_simulated_sensor_acquisition_processing(self) -> None:
        """Test 41: End-to-end integration: SimulatedSensor -> Acquisition -> Processor."""
        sensor = SimulatedThermalSensor(sensor_id="sim_core_0", start_temp=40.0, noise_std=0.0, seed=42)
        acq = ThermalDataAcquisition()

        raw_reading = acq.read_from_sensor(sensor)
        self.assertIsNotNone(raw_reading)

        processed = self.processor.process(raw_reading)
        self.assertEqual(processed.raw_reading.sensor_id, "sim_core_0")
        self.assertAlmostEqual(processed.raw_reading.temperature, 40.0, places=1)

    def test_42_invalid_acquisition_data_rejection(self) -> None:
        """Test 42: Processor rejection of invalid reading objects."""
        with self.assertRaises(InvalidReadingError):
            self.processor.process("not_a_reading")  # type: ignore

    def test_43_sensor_failure_behavior(self) -> None:
        """Test 43: Processor behavior when sensor acquisition fails."""
        # Simulated acquisition failure yielding None
        acq = ThermalDataAcquisition()
        reading = acq.read_from_sensor(None)  # returns None safely
        self.assertIsNone(reading)

        # Attempting to process None raises InvalidReadingError
        with self.assertRaises(InvalidReadingError):
            self.processor.process(reading)  # type: ignore

    def test_44_multiple_sequential_readings_multi_sensor_isolation(self) -> None:
        """Test 44: Multi-sensor isolation (Sensor A vs Sensor B history/statistics never mix)."""
        # Sensor A: 50°C -> 60°C (HEATING)
        self.processor.process(self._create_reading(50.0, seconds_offset=0, sensor_id="sensor_A"))
        self.processor.process(self._create_reading(60.0, seconds_offset=5, sensor_id="sensor_A"))

        # Sensor B: 90°C -> 80°C (COOLING)
        self.processor.process(self._create_reading(90.0, seconds_offset=0, sensor_id="sensor_B"))
        self.processor.process(self._create_reading(80.0, seconds_offset=5, sensor_id="sensor_B"))

        self.assertEqual(self.processor.get_trend("sensor_A"), ThermalTrend.HEATING)
        self.assertEqual(self.processor.get_trend("sensor_B"), ThermalTrend.COOLING)

        self.assertEqual(self.processor.get_average_temperature("sensor_A"), 55.0)
        self.assertEqual(self.processor.get_average_temperature("sensor_B"), 85.0)


if __name__ == "__main__":
    unittest.main()
