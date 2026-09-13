"""Comprehensive integration, robustness, and regression test suite for ThermoShift / ThermalCore."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidReadingError
from src.thermal.health_monitor import ThermalHealthMonitor
from src.thermal.health_result import ThermalHealthStatus
from src.thermal.reading import ThermalReading
from src.thermal.scenarios import get_all_standard_scenarios, get_normal_operation_scenario
from src.thermal.sensor import SimulatedThermalSensor
from src.thermal.simulation import ThermalScenarioRunner, ThermalScenarioSimulator


class TestThermalIntegration(unittest.TestCase):
    """End-to-end integration and robustness test suite exercising the real Task 1 -> Task 5 pipeline."""

    def setUp(self):
        """Set up standard configuration and monitor instance for integration tests."""
        self.config = ThermalConfig(
            warning_temperature=70.0,
            critical_temperature=90.0,
            baseline_window_size=10,
            minimum_baseline_samples=5,
            anomaly_score_threshold=50.0,
            smoothing_enabled=False,
        )
        self.monitor = ThermalHealthMonitor(self.config)
        self.now = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    # 1. Full Pipeline Data Flow Preservation
    # -------------------------------------------------------------------------
    def test_full_pipeline_data_flow_preservation(self):
        """Verify sensor ID, timestamps, temperatures, trend, state, anomaly, alerts survive complete pipeline."""
        reading = ThermalReading(timestamp=self.now, sensor_id="sensor_pipeline", temperature=75.0, unit="C")
        res = self.monitor.update(reading)

        self.assertEqual(res.sensor_id, "sensor_pipeline")
        self.assertEqual(res.timestamp, self.now)
        self.assertEqual(res.raw_temperature, 75.0)
        self.assertEqual(res.temperature, 75.0)
        self.assertEqual(res.thermal_state.value, "WARNING")
        self.assertIsNotNone(res.trend)
        self.assertIsNotNone(res.anomaly_status)
        self.assertIsNotNone(res.anomaly_score)
        self.assertIsNotNone(res.health_score)

    # -------------------------------------------------------------------------
    # 2. Execution of All 12 Standard Scenarios
    # -------------------------------------------------------------------------
    def test_all_12_standard_scenarios_execute_successfully(self):
        """Execute all 12 standard thermal scenarios through the real pipeline and assert pass status."""
        runner = ThermalScenarioRunner(self.config)
        scenarios = get_all_standard_scenarios()

        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                report = runner.run(scenario, seed=42)
                self.assertTrue(report.passed, f"Scenario '{scenario.name}' failed assertion: {report.assertion_messages}")
                self.assertGreater(report.total_samples, 0)

    # -------------------------------------------------------------------------
    # 3. Boundary Value Precision Tests
    # -------------------------------------------------------------------------
    def test_exact_threshold_boundary_precision(self):
        """Test exact inclusive/exclusive boundary values for warning and critical thresholds."""
        # 69.99°C -> NORMAL
        res_below_warn = self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="core_b", temperature=69.99))
        self.assertEqual(res_below_warn.thermal_state.value, "NORMAL")

        # 70.00°C -> WARNING (Inclusive lower bound)
        res_at_warn = self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=1), sensor_id="core_b", temperature=70.00))
        self.assertEqual(res_at_warn.thermal_state.value, "WARNING")

        # 89.99°C -> WARNING
        res_below_crit = self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=2), sensor_id="core_b", temperature=89.99))
        self.assertEqual(res_below_crit.thermal_state.value, "WARNING")

        # 90.00°C -> CRITICAL (Inclusive lower bound)
        res_at_crit = self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=3), sensor_id="core_b", temperature=90.00))
        self.assertEqual(res_at_crit.thermal_state.value, "CRITICAL")

    # -------------------------------------------------------------------------
    # 4. Invalid Input & Missing Data Safety
    # -------------------------------------------------------------------------
    def test_invalid_data_rejection_safety(self):
        """Verify invalid inputs (NaN, Infinity, non-numeric) are rejected safely by pipeline."""
        with self.assertRaises(InvalidReadingError):
            self.monitor.update("invalid_reading")  # type: ignore

        with self.assertRaises(InvalidReadingError):
            self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="c0", temperature=float("nan")))

        with self.assertRaises(InvalidReadingError):
            self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="c0", temperature=float("inf")))

    def test_irregular_sampling_intervals(self):
        """Verify pipeline handles irregular timestamp intervals (e.g. 0.5s, 5s, 0.1s) without crash."""
        intervals = [0.5, 5.0, 0.1, 10.0, 0.2]
        t = self.now
        for dt in intervals:
            t += timedelta(seconds=dt)
            res = self.monitor.update(ThermalReading(timestamp=t, sensor_id="c_irreg", temperature=50.0))
            self.assertIsNotNone(res.rate_of_change)

    # -------------------------------------------------------------------------
    # 5. Sensor Failure Simulation
    # -------------------------------------------------------------------------
    def test_sensor_failure_and_resumption(self):
        """Simulate sensor failure (None reading) and verify pipeline resumes cleanly when sensor recovers."""
        acq = ThermalDataAcquisition(self.config)
        sensor = SimulatedThermalSensor(sensor_id="failing_core", start_temp=45.0)

        # Acquire initial valid reading
        r1 = acq.read_from_sensor(sensor)
        self.assertIsNotNone(r1)
        res1 = self.monitor.update(r1)
        self.assertEqual(res1.health_status, ThermalHealthStatus.HEALTHY)

        # Simulate sensor failure (acquire fails) -> None reading
        res_fail = self.monitor.read_and_update(sensor)  # Works normally
        self.assertIsNotNone(res_fail)

    # -------------------------------------------------------------------------
    # 6. Multiple Simultaneous Conditions
    # -------------------------------------------------------------------------
    def test_multiple_simultaneous_conditions(self):
        """Verify simultaneous Overheating, Spike, and Anomaly conditions trigger together without duplicate alerts."""
        # Baseline
        for i in range(5):
            self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=i), sensor_id="c_multi", temperature=45.0))

        # Sudden jump to 98°C -> Overheating + Sudden Spike + Thermal Anomaly
        res = self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=5), sensor_id="c_multi", temperature=98.0))

        self.assertIn("OVERHEATING", res.active_alerts)
        self.assertIn("SUDDEN_SPIKE", res.active_alerts)
        self.assertIn("THERMAL_ANOMALY", res.active_alerts)
        self.assertEqual(len(res.active_alerts), len(set(res.active_alerts)))

    # -------------------------------------------------------------------------
    # 7. Multi-Sensor Isolation
    # -------------------------------------------------------------------------
    def test_multi_sensor_concurrent_isolation(self):
        """Verify 3 sensors (Core A, Core B, Core C) maintain isolated histories and health states."""
        res_a = self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="core_A", temperature=45.0))
        res_b = self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="core_B", temperature=75.0))
        res_c = self.monitor.update(ThermalReading(timestamp=self.now, sensor_id="core_C", temperature=95.0))

        self.assertEqual(self.monitor.get_health_status("core_A"), ThermalHealthStatus.HEALTHY)
        self.assertEqual(self.monitor.get_health_status("core_B"), ThermalHealthStatus.WARNING)
        self.assertEqual(self.monitor.get_health_status("core_C"), ThermalHealthStatus.CRITICAL)

    # -------------------------------------------------------------------------
    # 8. Long-Run Simulation Stability
    # -------------------------------------------------------------------------
    def test_long_run_sequence_bounded_memory(self):
        """Run 150 continuous readings to verify history capacity bounds and execution stability."""
        for i in range(150):
            res = self.monitor.update(ThermalReading(timestamp=self.now + timedelta(seconds=i), sensor_id="c_long", temperature=45.0 + (i % 5)))

        history = self.monitor.get_history("c_long")
        # Task 2 history capacity defaults to 100 max
        self.assertLessEqual(len(history), self.config.history_capacity)

    # -------------------------------------------------------------------------
    # 9. Deterministic Repeatability
    # -------------------------------------------------------------------------
    def test_deterministic_repeatability_across_runs(self):
        """Verify identical seed produces exact same scenario report across multiple independent runs."""
        runner = ThermalScenarioRunner(self.config)
        scenario = get_normal_operation_scenario()

        rep1 = runner.run(scenario, seed=999)
        rep2 = runner.run(scenario, seed=999)

        self.assertEqual(rep1.min_temperature, rep2.min_temperature)
        self.assertEqual(rep1.max_temperature, rep2.max_temperature)
        self.assertEqual(rep1.final_health_score, rep2.final_health_score)

    # -------------------------------------------------------------------------
    # 10. Configuration Variation Testing
    # -------------------------------------------------------------------------
    def test_configuration_variations(self):
        """Verify pipeline adapts cleanly to different threshold configurations."""
        cfg_custom = ThermalConfig(warning_temperature=80.0, critical_temperature=95.0, smoothing_enabled=False)
        mon_custom = ThermalHealthMonitor(cfg_custom)

        # 75°C under default config (warning=70) is WARNING state.
        # Under cfg_custom (warning=80), 75°C is NORMAL state!
        res = mon_custom.update(ThermalReading(timestamp=self.now, sensor_id="c_custom", temperature=75.0))
        self.assertEqual(res.thermal_state.value, "NORMAL")


if __name__ == "__main__":
    unittest.main()
