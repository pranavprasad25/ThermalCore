"""Unit tests for Thermal Safety subsystem configuration, monitoring, boundaries, and transitions."""

import math
import unittest

from src.safety.config import ThermalSafetyConfig
from src.safety.exceptions import InvalidThermalSafetyParameterError
from src.safety.monitor import ThermalSafetyMonitor
from src.safety.status import ThermalStatus, ThermalTransitionEventType


class TestThermalSafety(unittest.TestCase):
    """Test ThermalSafetyConfig, ThermalSafetyMonitor single-temperature checks, bounds, and hysteresis."""

    def setUp(self) -> None:
        """Initialize standard ThermalSafetyMonitor with default limits (70, 85, 95)."""
        self.config = ThermalSafetyConfig(
            warning_temperature=70.0,
            critical_temperature=85.0,
            maximum_temperature=95.0,
            hysteresis=0.0,
        )
        self.monitor = ThermalSafetyMonitor(config=self.config)

    # -------------------------------------------------------------------------
    # 1. Configuration & Validation Tests
    # -------------------------------------------------------------------------

    def test_valid_configuration(self) -> None:
        """Test instantiation of valid ThermalSafetyConfig."""
        self.assertEqual(self.config.warning_temperature, 70.0)
        self.assertEqual(self.config.critical_temperature, 85.0)
        self.assertEqual(self.config.maximum_temperature, 95.0)
        self.assertEqual(self.config.overheating_temperature, 95.0)

    def test_invalid_threshold_ordering(self) -> None:
        """Test that invalid threshold ordering raises InvalidThermalSafetyParameterError."""
        # warning >= critical
        with self.assertRaises(InvalidThermalSafetyParameterError):
            ThermalSafetyConfig(warning_temperature=85.0, critical_temperature=85.0, maximum_temperature=95.0)

        # critical >= maximum
        with self.assertRaises(InvalidThermalSafetyParameterError):
            ThermalSafetyConfig(warning_temperature=70.0, critical_temperature=95.0, maximum_temperature=90.0)

    def test_non_finite_thresholds(self) -> None:
        """Test NaN or Inf thresholds raise InvalidThermalSafetyParameterError."""
        with self.assertRaises(InvalidThermalSafetyParameterError):
            ThermalSafetyConfig(warning_temperature=float("nan"))

        with self.assertRaises(InvalidThermalSafetyParameterError):
            ThermalSafetyConfig(critical_temperature=float("inf"))

    def test_invalid_temperature_check_inputs(self) -> None:
        """Test NaN, Inf, or out-of-bounds inputs raise InvalidThermalSafetyParameterError."""
        with self.assertRaises(InvalidThermalSafetyParameterError):
            self.monitor.check_temperature(float("nan"))

        with self.assertRaises(InvalidThermalSafetyParameterError):
            self.monitor.check_temperature(float("inf"))

        with self.assertRaises(InvalidThermalSafetyParameterError):
            self.monitor.check_temperature(500.0)  # out of physical bounds

    # -------------------------------------------------------------------------
    # 2. State & Boundary Semantics Tests
    # -------------------------------------------------------------------------

    def test_normal_state_and_safety_semantics(self) -> None:
        """Test NORMAL state classification and verify is_safe is True ONLY when NORMAL."""
        res_low = self.monitor.check_temperature(50.0)
        self.assertEqual(res_low.status, ThermalStatus.NORMAL)
        self.assertTrue(res_low.is_safe)
        self.assertFalse(res_low.is_warning)
        self.assertFalse(res_low.is_critical)
        self.assertFalse(res_low.is_overheating)

        # Exactly below warning threshold (69.9 °C)
        res_boundary = self.monitor.check_temperature(69.9)
        self.assertEqual(res_boundary.status, ThermalStatus.NORMAL)
        self.assertTrue(res_boundary.is_safe)

    def test_warning_state_semantics(self) -> None:
        """Test WARNING state classification for warning_temp <= T < critical_temp."""
        # Exact warning threshold (70.0 °C)
        res_exact = self.monitor.check_temperature(70.0)
        self.assertEqual(res_exact.status, ThermalStatus.WARNING)
        self.assertFalse(res_exact.is_safe)  # is_safe MUST be False
        self.assertTrue(res_exact.is_warning)

        # Mid warning range (80.0 °C)
        res_mid = self.monitor.check_temperature(80.0)
        self.assertEqual(res_mid.status, ThermalStatus.WARNING)

        # Exactly below critical threshold (84.9 °C)
        res_top = self.monitor.check_temperature(84.9)
        self.assertEqual(res_top.status, ThermalStatus.WARNING)

    def test_critical_state_semantics(self) -> None:
        """Test CRITICAL state classification for critical_temp <= T < maximum_temp."""
        # Exact critical threshold (85.0 °C)
        res_exact = self.monitor.check_temperature(85.0)
        self.assertEqual(res_exact.status, ThermalStatus.CRITICAL)
        self.assertFalse(res_exact.is_safe)
        self.assertTrue(res_exact.is_critical)

        # Mid critical range (90.0 °C)
        res_mid = self.monitor.check_temperature(90.0)
        self.assertEqual(res_mid.status, ThermalStatus.CRITICAL)

        # Below maximum threshold (94.9 °C)
        res_top = self.monitor.check_temperature(94.9)
        self.assertEqual(res_top.status, ThermalStatus.CRITICAL)

    def test_overheating_state_semantics(self) -> None:
        """Test OVERHEATING state classification for T >= maximum_temp."""
        # Exact maximum threshold (95.0 °C)
        res_exact = self.monitor.check_temperature(95.0)
        self.assertEqual(res_exact.status, ThermalStatus.OVERHEATING)
        self.assertFalse(res_exact.is_safe)
        self.assertTrue(res_exact.is_overheating)
        self.assertTrue(self.monitor.is_overheating(95.0))

        # Above maximum threshold (105.0 °C)
        res_high = self.monitor.check_temperature(105.0)
        self.assertEqual(res_high.status, ThermalStatus.OVERHEATING)
        self.assertTrue(res_high.is_overheating)

    # -------------------------------------------------------------------------
    # 3. Hysteresis & Transition Tests
    # -------------------------------------------------------------------------

    def test_hysteresis_flapping_prevention(self) -> None:
        """Test that non-zero hysteresis prevents rapid state de-escalation around boundaries."""
        h_monitor = ThermalSafetyMonitor(
            warning_temperature=70.0,
            critical_temperature=85.0,
            maximum_temperature=95.0,
            hysteresis=2.0,  # 2 °C hysteresis offset
        )

        # Step 1: Rise to 85.5 °C -> CRITICAL
        res1 = h_monitor.check_temperature(85.5)
        self.assertEqual(res1.status, ThermalStatus.CRITICAL)

        # Step 2: Drop to 84.5 °C (below 85 °C, but above 85 - 2 = 83 °C hysteresis limit)
        # Should REMAIN in CRITICAL state due to hysteresis
        res2 = h_monitor.check_temperature(84.5, previous_status=res1.status)
        self.assertEqual(res2.status, ThermalStatus.CRITICAL)

        # Step 3: Drop to 82.5 °C (below 83 °C) -> De-escalates to WARNING
        res3 = h_monitor.check_temperature(82.5, previous_status=res2.status)
        self.assertEqual(res3.status, ThermalStatus.WARNING)


if __name__ == "__main__":
    unittest.main()
