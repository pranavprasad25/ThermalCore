"""Unit tests for TemperatureMonitor per-core thermal tracking subsystem."""

from datetime import datetime, timezone
import math
import unittest

from src.thermal.exceptions import InvalidReadingError
from src.thermal.temperature_monitor import CoreThermalState, TemperatureMonitor


class TestTemperatureMonitor(unittest.TestCase):
    """Unit test suite for TemperatureMonitor covering all Day 4 Task 1 requirements."""

    def setUp(self) -> None:
        """Instantiate clean TemperatureMonitor before each test."""
        self.monitor = TemperatureMonitor(history_size=5, min_valid_temp=-50.0, max_valid_temp=150.0)

    # -------------------------------------------------------------------------
    # Test 1: First Temperature Reading
    # -------------------------------------------------------------------------
    def test_01_first_temperature_reading(self) -> None:
        """Test 1: Verify first reading initializes state with current=temp, previous=None, difference=None."""
        state = self.monitor.update_temperature(core_id=0, temperature=75.0, timestamp=10.0)

        self.assertEqual(state.core_id, 0)
        self.assertEqual(state.current_temperature, 75.0)
        self.assertIsNone(state.previous_temperature)
        self.assertIsNone(state.temperature_difference)
        self.assertEqual(state.current_timestamp, 10.0)
        self.assertIsNone(state.previous_timestamp)
        self.assertIsNone(state.time_difference)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0], (10.0, 75.0))

        # Check getter methods
        self.assertEqual(self.monitor.get_current_temperature(0), 75.0)
        self.assertIsNone(self.monitor.get_previous_temperature(0))
        self.assertIsNone(self.monitor.get_temperature_difference(0))
        self.assertIsNone(self.monitor.get_time_difference(0))

    # -------------------------------------------------------------------------
    # Test 2: Second Temperature Reading (Temperature Rise)
    # -------------------------------------------------------------------------
    def test_02_second_temperature_reading_rise(self) -> None:
        """Test 2: Verify second reading updates current, previous, delta T, and delta t correctly."""
        self.monitor.update_temperature(core_id=0, temperature=70.0, timestamp=0.0)
        state = self.monitor.update_temperature(core_id=0, temperature=75.0, timestamp=1.0)

        self.assertEqual(state.current_temperature, 75.0)
        self.assertEqual(state.previous_temperature, 70.0)
        self.assertEqual(state.temperature_difference, 5.0)
        self.assertEqual(state.current_timestamp, 1.0)
        self.assertEqual(state.previous_timestamp, 0.0)
        self.assertEqual(state.time_difference, 1.0)
        self.assertEqual(len(state.history), 2)

        # Getters
        self.assertEqual(self.monitor.get_current_temperature(0), 75.0)
        self.assertEqual(self.monitor.get_previous_temperature(0), 70.0)
        self.assertEqual(self.monitor.get_temperature_difference(0), 5.0)
        self.assertEqual(self.monitor.get_time_difference(0), 1.0)

    # -------------------------------------------------------------------------
    # Test 3: Temperature Decrease
    # -------------------------------------------------------------------------
    def test_03_temperature_decrease(self) -> None:
        """Test 3: Verify temperature decrease results in negative temperature difference."""
        self.monitor.update_temperature(core_id="core_0", temperature=75.0, timestamp=5.0)
        state = self.monitor.update_temperature(core_id="core_0", temperature=70.0, timestamp=7.0)

        self.assertEqual(state.current_temperature, 70.0)
        self.assertEqual(state.previous_temperature, 75.0)
        self.assertEqual(state.temperature_difference, -5.0)
        self.assertEqual(state.time_difference, 2.0)

    # -------------------------------------------------------------------------
    # Test 4: Stable Temperature
    # -------------------------------------------------------------------------
    def test_04_stable_temperature(self) -> None:
        """Test 4: Verify identical consecutive temperature readings yield zero difference."""
        self.monitor.update_temperature(core_id=1, temperature=75.0, timestamp=0.0)
        state = self.monitor.update_temperature(core_id=1, temperature=75.0, timestamp=1.0)

        self.assertEqual(state.current_temperature, 75.0)
        self.assertEqual(state.previous_temperature, 75.0)
        self.assertEqual(state.temperature_difference, 0.0)

    # -------------------------------------------------------------------------
    # Test 5: Multiple Cores Data Isolation
    # -------------------------------------------------------------------------
    def test_05_multiple_cores_isolation(self) -> None:
        """Test 5: Verify updating one core does not alter or corrupt other cores' states or histories."""
        self.monitor.update_temperature(core_id=0, temperature=65.0, timestamp=0.0)
        self.monitor.update_temperature(core_id=1, temperature=81.0, timestamp=0.0)
        self.monitor.update_temperature(core_id=2, temperature=72.0, timestamp=0.0)
        self.monitor.update_temperature(core_id=3, temperature=68.0, timestamp=0.0)

        # Update Core 2 only
        self.monitor.update_temperature(core_id=2, temperature=77.0, timestamp=1.0)

        # Verify Core 2 updated
        self.assertEqual(self.monitor.get_current_temperature(2), 77.0)
        self.assertEqual(self.monitor.get_previous_temperature(2), 72.0)
        self.assertEqual(self.monitor.get_temperature_difference(2), 5.0)

        # Verify Core 0, 1, 3 remain unaffected
        self.assertEqual(self.monitor.get_current_temperature(0), 65.0)
        self.assertIsNone(self.monitor.get_previous_temperature(0))

        self.assertEqual(self.monitor.get_current_temperature(1), 81.0)
        self.assertIsNone(self.monitor.get_previous_temperature(1))

        self.assertEqual(self.monitor.get_current_temperature(3), 68.0)
        self.assertIsNone(self.monitor.get_previous_temperature(3))

        # Check all temperatures map
        all_temps = self.monitor.get_all_core_temperatures()
        self.assertEqual(all_temps, {0: 65.0, 1: 81.0, 2: 77.0, 3: 68.0})

    # -------------------------------------------------------------------------
    # Test 6: History Size Limit Windowing
    # -------------------------------------------------------------------------
    def test_06_history_size_limit_windowing(self) -> None:
        """Test 6: Verify history buffer rolls off oldest entries when max capacity (5) is reached."""
        for i in range(10):
            self.monitor.update_temperature(core_id="CPU_A", temperature=50.0 + i, timestamp=float(i))

        history = self.monitor.get_temperature_history("CPU_A")
        self.assertEqual(len(history), 5)
        # Oldest preserved reading should be i=5 (55.0°C) and newest i=9 (59.0°C)
        self.assertEqual(history[0], (5.0, 55.0))
        self.assertEqual(history[-1], (9.0, 59.0))

    # -------------------------------------------------------------------------
    # Test 7: Invalid Temperature Rejections
    # -------------------------------------------------------------------------
    def test_07_invalid_temperature_rejections(self) -> None:
        """Test 7: Verify invalid temperatures (None, str, list, dict, bool, NaN, Inf) are rejected with exception."""
        invalid_temps = [
            None,
            "hot",
            "75.0",  # String non-coercible or type check
            [75.0],
            {"temp": 75.0},
            True,
            False,
            float("nan"),
            float("inf"),
            float("-inf"),
            200.0,  # Exceeds max_valid_temp (150.0)
            -100.0, # Below min_valid_temp (-50.0)
        ]

        for inv_t in invalid_temps:
            with self.subTest(invalid_temp=inv_t):
                with self.assertRaises(InvalidReadingError):
                    self.monitor.update_temperature(core_id=0, temperature=inv_t)  # type: ignore

    # -------------------------------------------------------------------------
    # Test 8: Negative Celsius Temperature Handling
    # -------------------------------------------------------------------------
    def test_08_negative_celsius_temperature_handling(self) -> None:
        """Test 8: Verify valid negative Celsius temperatures (e.g. -10°C, -25.5°C) are processed correctly."""
        state1 = self.monitor.update_temperature(core_id=0, temperature=-10.0, timestamp=0.0)
        self.assertEqual(state1.current_temperature, -10.0)

        state2 = self.monitor.update_temperature(core_id=0, temperature=-25.5, timestamp=1.0)
        self.assertEqual(state2.current_temperature, -25.5)
        self.assertEqual(state2.previous_temperature, -10.0)
        self.assertEqual(state2.temperature_difference, -15.5)

    # -------------------------------------------------------------------------
    # Test 9: Timestamps and Irregular Sampling Intervals
    # -------------------------------------------------------------------------
    def test_09_timestamps_and_sampling_intervals(self) -> None:
        """Test 9: Verify irregular sampling intervals (t=0s, 1s, 3.5s, 10s) compute accurate time differences."""
        r1 = self.monitor.update_temperature(core_id=0, temperature=70.0, timestamp=0.0)
        self.assertIsNone(r1.time_difference)

        r2 = self.monitor.update_temperature(core_id=0, temperature=72.0, timestamp=1.0)
        self.assertEqual(r2.time_difference, 1.0)

        r3 = self.monitor.update_temperature(core_id=0, temperature=75.0, timestamp=3.5)
        self.assertEqual(r3.time_difference, 2.5)

        r4 = self.monitor.update_temperature(core_id=0, temperature=80.0, timestamp=10.0)
        self.assertEqual(r4.time_difference, 6.5)

        # Datetime timestamp objects support
        dt1 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 9, 17, 12, 0, 5, tzinfo=timezone.utc)

        self.monitor.update_temperature(core_id=1, temperature=50.0, timestamp=dt1)
        r_dt = self.monitor.update_temperature(core_id=1, temperature=55.0, timestamp=dt2)
        self.assertEqual(r_dt.time_difference, 5.0)

    # -------------------------------------------------------------------------
    # Test 10: Sudden Temperature Jump Safety
    # -------------------------------------------------------------------------
    def test_10_sudden_temperature_jump_safety(self) -> None:
        """Test 10: Verify sudden temperature jump (70°C -> 72°C -> 95°C) is recorded safely without crash."""
        self.monitor.update_temperature(core_id=0, temperature=70.0, timestamp=0.0)
        self.monitor.update_temperature(core_id=0, temperature=72.0, timestamp=1.0)
        state_jump = self.monitor.update_temperature(core_id=0, temperature=95.0, timestamp=2.0)

        self.assertEqual(state_jump.current_temperature, 95.0)
        self.assertEqual(state_jump.previous_temperature, 72.0)
        self.assertEqual(state_jump.temperature_difference, 23.0)
        self.assertEqual(len(state_jump.history), 3)

    # -------------------------------------------------------------------------
    # Test 11: Unknown Core Getter Behavior
    # -------------------------------------------------------------------------
    def test_11_unknown_core_getter_behavior(self) -> None:
        """Test 11: Verify getters return None or empty list for unknown core IDs without raising exceptions."""
        self.assertIsNone(self.monitor.get_current_temperature("non_existent_core"))
        self.assertIsNone(self.monitor.get_previous_temperature(999))
        self.assertIsNone(self.monitor.get_temperature_difference("core_X"))
        self.assertIsNone(self.monitor.get_time_difference("core_X"))
        self.assertIsNone(self.monitor.get_latest_reading("core_X"))
        self.assertEqual(self.monitor.get_temperature_history("core_X"), [])

    # -------------------------------------------------------------------------
    # Test 12: Empty History & Core Reset Behavior
    # -------------------------------------------------------------------------
    def test_12_empty_history_and_reset(self) -> None:
        """Test 12: Verify empty monitor state and resetting individual or all core records."""
        self.assertEqual(self.monitor.get_all_core_temperatures(), {})
        self.assertEqual(self.monitor.get_monitored_cores(), [])

        self.monitor.update_temperature(core_id=0, temperature=60.0)
        self.monitor.update_temperature(core_id=1, temperature=65.0)

        self.assertEqual(len(self.monitor.get_monitored_cores()), 2)

        # Reset core 0
        self.monitor.reset(core_id=0)
        self.assertIsNone(self.monitor.get_current_temperature(0))
        self.assertEqual(self.monitor.get_current_temperature(1), 65.0)

        # Reset all
        self.monitor.reset()
        self.assertEqual(self.monitor.get_all_core_temperatures(), {})

    # -------------------------------------------------------------------------
    # Test 13: Core ID Validation Errors
    # -------------------------------------------------------------------------
    def test_13_invalid_core_id_rejections(self) -> None:
        """Test 13: Verify invalid core IDs (None, empty string, negative int, invalid type) raise InvalidReadingError."""
        invalid_core_ids = [
            None,
            "",
            "   ",
            -1,
            -10,
            True,
            False,
            [0],
            {"id": 0},
        ]

        for inv_id in invalid_core_ids:
            with self.subTest(invalid_core_id=inv_id):
                with self.assertRaises(InvalidReadingError):
                    self.monitor.update_temperature(core_id=inv_id, temperature=65.0)  # type: ignore


if __name__ == "__main__":
    unittest.main()
