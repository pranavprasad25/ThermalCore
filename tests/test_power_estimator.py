"""Automated unit test suite for Day 3 — Task 1: Power Estimation."""

from datetime import datetime, timezone
import math
import unittest

from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.power.exceptions import (
    InvalidPowerParameterError,
    PowerError,
    PowerEstimationError,
)
from src.power.power_result import PowerResult


class TestPowerEstimation(unittest.TestCase):
    """Unit tests covering physical power models, equations, configuration, and error validation."""

    def setUp(self) -> None:
        """Set up test fixture with standard reference configuration."""
        self.config = PowerConfig(
            capacitance=1.5e-9,  # 1.5 nF
            base_static_power=5.0,  # 5.0 W
            activity_factor_scale=1.0,
            base_activity_factor=0.0,
            min_voltage=0.5,
            max_voltage=2.0,
            min_frequency=5.0e8,  # 500 MHz
            max_frequency=6.0e9,  # 6.0 GHz
            reference_voltage=1.0,
            reference_temperature=25.0,
            temperature_coefficient=0.015,
            static_power_model="CONSTANT",
        )
        self.estimator = PowerEstimator(config=self.config)

    # --------------------------------------------------------------------------
    # 1. Dynamic Power Calculation
    # --------------------------------------------------------------------------

    def test_dynamic_power_calculation_exact(self) -> None:
        """Test dynamic power equation: P_dynamic = α × C × V² × f with manual hand calculation."""
        # Workload = 0.8 -> α = 0.8
        # C = 1.5e-9 F, V = 1.1 V, f = 3.0e9 Hz (3 GHz)
        # V² = 1.21
        # P_dyn = 0.8 * 1.5e-9 * 1.21 * 3.0e9 = 0.8 * 1.5 * 1.21 * 3.0 = 4.356 W
        p_dyn = self.estimator.estimate_dynamic_power(
            workload=0.8,
            voltage=1.1,
            frequency=3.0e9,
        )
        self.assertAlmostEqual(p_dyn, 4.356, places=6)

    def test_dynamic_power_with_explicit_activity_factor(self) -> None:
        """Test dynamic power with directly provided activity factor."""
        p_dyn = self.estimator.estimate_dynamic_power(
            workload=0.5,
            voltage=1.2,
            frequency=2.5e9,
            activity_factor=0.75,
        )
        # P = 0.75 * 1.5e-9 * (1.2^2) * 2.5e9 = 0.75 * 1.5 * 1.44 * 2.5 = 4.05 W
        self.assertAlmostEqual(p_dyn, 4.05, places=6)

    def test_dynamic_power_with_capacitance_override(self) -> None:
        """Test dynamic power with custom capacitance passed to method."""
        p_dyn = self.estimator.estimate_dynamic_power(
            workload=1.0,
            voltage=1.0,
            frequency=2.0e9,
            capacitance=2.0e-9,  # 2.0 nF
        )
        # P = 1.0 * 2.0e-9 * 1.0^2 * 2.0e9 = 4.0 W
        self.assertAlmostEqual(p_dyn, 4.0, places=6)

    # --------------------------------------------------------------------------
    # 2. Static Power Calculation
    # --------------------------------------------------------------------------

    def test_static_power_constant_baseline(self) -> None:
        """Test constant baseline static power model."""
        p_static = self.estimator.estimate_static_power()
        self.assertEqual(p_static, 5.0)

    def test_static_power_linear_temperature_model(self) -> None:
        """Test temperature-dependent linear static power scaling."""
        linear_config = PowerConfig(
            base_static_power=5.0,
            reference_temperature=25.0,
            temperature_coefficient=0.02,
            static_power_model="LINEAR",
        )
        estimator = PowerEstimator(config=linear_config)

        # At T = 25.0 °C (reference): P = 5.0 * (1 + 0.02 * 0) = 5.0 W
        self.assertAlmostEqual(estimator.estimate_static_power(temperature=25.0), 5.0, places=6)

        # At T = 75.0 °C: ΔT = 50.0 °C -> factor = 1 + 0.02*50 = 2.0 -> P = 10.0 W
        self.assertAlmostEqual(estimator.estimate_static_power(temperature=75.0), 10.0, places=6)

    def test_static_power_exponential_temperature_model(self) -> None:
        """Test temperature-dependent exponential static power scaling."""
        exp_config = PowerConfig(
            base_static_power=4.0,
            reference_temperature=30.0,
            temperature_coefficient=0.015,
            static_power_model="EXPONENTIAL",
        )
        estimator = PowerEstimator(config=exp_config)

        # At T = 30.0 °C: P = 4.0 * exp(0) = 4.0 W
        self.assertAlmostEqual(estimator.estimate_static_power(temperature=30.0), 4.0, places=6)

        # At T = 70.0 °C: ΔT = 40.0 °C -> P = 4.0 * exp(0.015 * 40) = 4.0 * exp(0.6) ≈ 7.288475 W
        expected = 4.0 * math.exp(0.6)
        self.assertAlmostEqual(estimator.estimate_static_power(temperature=70.0), expected, places=5)

    def test_static_power_voltage_dependent_leakage(self) -> None:
        """Test static power with voltage dependence enabled."""
        v_leak_config = PowerConfig(
            base_static_power=6.0,
            reference_voltage=1.0,
            voltage_dependent_leakage=True,
        )
        estimator = PowerEstimator(config=v_leak_config)

        # At V = 1.2 V -> P = 6.0 * (1.2 / 1.0) = 7.2 W
        self.assertAlmostEqual(estimator.estimate_static_power(voltage=1.2), 7.2, places=6)

    # --------------------------------------------------------------------------
    # 3. Total Power Calculation
    # --------------------------------------------------------------------------

    def test_total_power_sum(self) -> None:
        """Test that Total Power strictly equals Dynamic Power + Static Power."""
        p_dyn = self.estimator.estimate_dynamic_power(workload=0.8, voltage=1.1, frequency=3.0e9)
        p_stat = self.estimator.estimate_static_power()
        p_total = self.estimator.estimate_total_power(workload=0.8, voltage=1.1, frequency=3.0e9)

        self.assertAlmostEqual(p_dyn, 4.356, places=6)
        self.assertAlmostEqual(p_stat, 5.0, places=6)
        self.assertAlmostEqual(p_total, p_dyn + p_stat, places=6)
        self.assertAlmostEqual(p_total, 9.356, places=6)

    # --------------------------------------------------------------------------
    # 4. Zero / Low Workload Behavior
    # --------------------------------------------------------------------------

    def test_zero_workload_dynamic_power_is_zero(self) -> None:
        """Test that dynamic power is 0.0 W when workload is 0.0 (idle)."""
        res = self.estimator.estimate(workload=0.0, voltage=1.0, frequency=2.0e9)
        self.assertEqual(res.dynamic_power, 0.0)
        self.assertEqual(res.static_power, 5.0)
        self.assertEqual(res.total_power, 5.0)

    def test_zero_workload_with_base_activity(self) -> None:
        """Test idle power when base activity factor is non-zero."""
        config = PowerConfig(
            capacitance=1.0e-9,
            base_static_power=2.0,
            base_activity_factor=0.05,
            activity_factor_scale=0.95,
        )
        estimator = PowerEstimator(config=config)
        res = estimator.estimate(workload=0.0, voltage=1.0, frequency=1.0e9)
        # P_dyn = 0.05 * 1.0e-9 * 1.0^2 * 1.0e9 = 0.05 W
        self.assertAlmostEqual(res.dynamic_power, 0.05, places=6)
        self.assertAlmostEqual(res.total_power, 2.05, places=6)

    # --------------------------------------------------------------------------
    # 5. Workload Scaling Relationships
    # --------------------------------------------------------------------------

    def test_workload_linear_scaling(self) -> None:
        """Test that dynamic power scales linearly with workload utilization."""
        w_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        powers = [
            self.estimator.estimate_dynamic_power(workload=w, voltage=1.0, frequency=2.0e9)
            for w in w_levels
        ]

        # Check monotonic increase
        for i in range(len(powers) - 1):
            self.assertLess(powers[i], powers[i + 1])

        # Check linearity: power at 0.4 should be exactly 2x power at 0.2
        self.assertAlmostEqual(powers[1], 2.0 * powers[0], places=6)
        self.assertAlmostEqual(powers[4], 5.0 * powers[0], places=6)

    # --------------------------------------------------------------------------
    # 6. Frequency Scaling Relationships
    # --------------------------------------------------------------------------

    def test_frequency_linear_scaling(self) -> None:
        """Test that dynamic power scales linearly with clock frequency."""
        f1 = 1.0e9
        f2 = 2.0e9
        f3 = 4.0e9

        p1 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=f1)
        p2 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=f2)
        p3 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=f3)

        self.assertAlmostEqual(p2, 2.0 * p1, places=6)
        self.assertAlmostEqual(p3, 4.0 * p1, places=6)

    # --------------------------------------------------------------------------
    # 7. Voltage Quadratic (V²) Scaling Relationships
    # --------------------------------------------------------------------------

    def test_voltage_quadratic_scaling(self) -> None:
        """Test that dynamic power scales quadratically (V²) with voltage."""
        v1 = 0.8
        v2 = 1.2
        v3 = 1.6

        p1 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=v1, frequency=2.0e9)
        p2 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=v2, frequency=2.0e9)
        p3 = self.estimator.estimate_dynamic_power(workload=0.5, voltage=v3, frequency=2.0e9)

        # Ratio of p2 / p1 should be (1.2 / 0.8)^2 = 1.5^2 = 2.25
        self.assertAlmostEqual(p2 / p1, (v2 / v1) ** 2, places=6)
        # Ratio of p3 / p1 should be (1.6 / 0.8)^2 = 2.0^2 = 4.0
        self.assertAlmostEqual(p3 / p1, 4.0, places=6)

    # --------------------------------------------------------------------------
    # 8. Invalid Negative and Non-Finite Inputs
    # --------------------------------------------------------------------------

    def test_negative_voltage_raises_error(self) -> None:
        """Test that negative voltage raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=-1.0, frequency=2.0e9)

    def test_zero_voltage_raises_error(self) -> None:
        """Test that zero voltage raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=0.0, frequency=2.0e9)

    def test_negative_frequency_raises_error(self) -> None:
        """Test that negative frequency raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=1.0, frequency=-2.0e9)

    def test_negative_capacitance_raises_error(self) -> None:
        """Test that negative capacitance raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=2.0e9, capacitance=-1.0e-9)

    def test_negative_activity_factor_raises_error(self) -> None:
        """Test that negative activity factor raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=2.0e9, activity_factor=-0.5)

    def test_nan_and_inf_inputs_raise_error(self) -> None:
        """Test that NaN and Inf parameters are caught and rejected."""
        invalid_values = [float("nan"), float("inf"), float("-inf")]
        for val in invalid_values:
            with self.subTest(val=val):
                with self.assertRaises(InvalidPowerParameterError):
                    self.estimator.estimate(workload=val, voltage=1.0, frequency=2.0e9)
                with self.assertRaises(InvalidPowerParameterError):
                    self.estimator.estimate(workload=0.5, voltage=val, frequency=2.0e9)
                with self.assertRaises(InvalidPowerParameterError):
                    self.estimator.estimate(workload=0.5, voltage=1.0, frequency=val)

    # --------------------------------------------------------------------------
    # 9. Invalid Workload Range
    # --------------------------------------------------------------------------

    def test_workload_below_zero_raises_error(self) -> None:
        """Test that workload < 0.0 raises InvalidPowerParameterError."""
        with self.assertRaises(InvalidPowerParameterError) as ctx:
            self.estimator.estimate(workload=-0.05, voltage=1.0, frequency=2.0e9)
        self.assertIn("negative", str(ctx.exception).lower())

    def test_workload_above_one_raises_error_with_guidance(self) -> None:
        """Test that workload > 1.0 (e.g. 80%) raises InvalidPowerParameterError with helpful message."""
        with self.assertRaises(InvalidPowerParameterError) as ctx:
            self.estimator.estimate(workload=80.0, voltage=1.0, frequency=2.0e9)
        self.assertIn("percentage", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 10. Voltage & Frequency Operating Limits
    # --------------------------------------------------------------------------

    def test_voltage_out_of_bounds(self) -> None:
        """Test voltage below min_voltage or above max_voltage."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=0.2, frequency=2.0e9)  # < min_voltage (0.5 V)
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=2.5, frequency=2.0e9)  # > max_voltage (2.0 V)

    def test_frequency_out_of_bounds(self) -> None:
        """Test frequency below min_frequency or above max_frequency."""
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=1.0, frequency=1.0e8)  # < min_frequency (500 MHz)
        with self.assertRaises(InvalidPowerParameterError):
            self.estimator.estimate(workload=0.5, voltage=1.0, frequency=7.0e9)  # > max_frequency (6.0 GHz)

    # --------------------------------------------------------------------------
    # 11. Configuration Validation
    # --------------------------------------------------------------------------

    def test_power_config_validation(self) -> None:
        """Test that PowerConfig validates its fields in __post_init__."""
        with self.assertRaises(InvalidPowerParameterError):
            PowerConfig(capacitance=-1.0)
        with self.assertRaises(InvalidPowerParameterError):
            PowerConfig(base_static_power=-5.0)
        with self.assertRaises(InvalidPowerParameterError):
            PowerConfig(min_voltage=2.0, max_voltage=1.0)
        with self.assertRaises(InvalidPowerParameterError):
            PowerConfig(min_frequency=3.0e9, max_frequency=2.0e9)
        with self.assertRaises(InvalidPowerParameterError):
            PowerConfig(static_power_model="INVALID_MODEL")

    # --------------------------------------------------------------------------
    # 12. API Return Structure & Immutability
    # --------------------------------------------------------------------------

    def test_power_result_structure_and_serialization(self) -> None:
        """Test PowerResult properties, immutability, and to_dict() serialization."""
        now = datetime.now(timezone.utc)
        result = self.estimator.estimate(
            workload=0.75,
            voltage=1.15,
            frequency=3.2e9,
            temperature=68.5,
            metadata={"sensor_id": "core_0", "core_type": "performance"},
            timestamp=now,
        )

        self.assertIsInstance(result, PowerResult)
        self.assertEqual(result.workload, 0.75)
        self.assertEqual(result.voltage, 1.15)
        self.assertEqual(result.frequency, 3.2e9)
        self.assertEqual(result.temperature, 68.5)
        self.assertEqual(result.timestamp, now)
        self.assertEqual(result.metadata["sensor_id"], "core_0")

        # Verify immutability (frozen dataclass)
        with self.assertRaises(AttributeError):
            result.dynamic_power = 99.9  # type: ignore

        # Verify to_dict output
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("dynamic_power", d)
        self.assertIn("static_power", d)
        self.assertIn("total_power", d)
        self.assertEqual(d["frequency_ghz"], 3.2)
        self.assertEqual(d["temperature_c"], 68.5)
        self.assertEqual(d["timestamp"], now.isoformat())

    # --------------------------------------------------------------------------
    # 13. Batch Estimation
    # --------------------------------------------------------------------------

    def test_batch_estimation(self) -> None:
        """Test estimate_batch across multiple operating points."""
        points = [
            {"workload": 0.1, "voltage": 0.8, "frequency": 1.0e9},
            {"workload": 0.5, "voltage": 1.0, "frequency": 2.0e9},
            {"workload": 1.0, "voltage": 1.2, "frequency": 3.0e9},
        ]
        results = self.estimator.estimate_batch(points)
        self.assertEqual(len(results), 3)
        self.assertLess(results[0].dynamic_power, results[1].dynamic_power)
        self.assertLess(results[1].dynamic_power, results[2].dynamic_power)

    def test_batch_estimation_error_wrapping(self) -> None:
        """Test batch estimation error handling with invalid point."""
        points = [
            {"workload": 0.5, "voltage": 1.0, "frequency": 2.0e9},
            {"workload": -0.5, "voltage": 1.0, "frequency": 2.0e9},  # Invalid!
        ]
        with self.assertRaises(PowerEstimationError):
            self.estimator.estimate_batch(points)


if __name__ == "__main__":
    unittest.main()
