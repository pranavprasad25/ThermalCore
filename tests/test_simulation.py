"""Automated unit test suite for thermal simulation framework and scenario generators."""

from datetime import datetime, timezone
import math
import unittest


from src.thermal.scenarios import (
    get_all_standard_scenarios,
    get_normal_operation_scenario,
    get_overheating_scenario,
    get_sudden_spike_scenario,
)
from src.thermal.simulation import (
    ThermalScenario,
    ThermalScenarioPhase,
    ThermalScenarioReport,
    ThermalScenarioRunner,
    ThermalScenarioSimulator,
)


class TestThermalSimulation(unittest.TestCase):
    """Test suite covering thermal scenario simulation framework and standard scenarios."""

    def test_simulator_deterministic_reproducibility(self):
        """Verify fixed random seed produces identical reading sequences across calls."""
        scenario = get_normal_operation_scenario()

        sim1 = ThermalScenarioSimulator(seed=12345)
        readings1 = sim1.generate_readings(scenario)

        sim2 = ThermalScenarioSimulator(seed=12345)
        readings2 = sim2.generate_readings(scenario)

        self.assertEqual(len(readings1), len(readings2))
        for r1, r2 in zip(readings1, readings2):
            self.assertEqual(r1.temperature, r2.temperature)
            self.assertEqual(r1.sensor_id, r2.sensor_id)

    def test_simulator_generated_reading_structure(self):
        """Verify generated readings have correct timestamps, units, and non-empty values."""
        scenario = get_normal_operation_scenario()
        sim = ThermalScenarioSimulator(seed=42)
        readings = sim.generate_readings(scenario)

        self.assertGreater(len(readings), 0)
        for r in readings:
            self.assertEqual(r.sensor_id, scenario.sensor_id)
            self.assertEqual(r.unit, "C")
            self.assertIsNotNone(r.timestamp)
            self.assertFalse(math.isnan(r.temperature) if hasattr(math, "isnan") else False)

    def test_all_12_standard_scenarios_exist(self):
        """Verify all 12 required standard scenarios are instantiated by get_all_standard_scenarios."""
        scenarios = get_all_standard_scenarios()
        self.assertEqual(len(scenarios), 12)
        names = [s.name for s in scenarios]
        self.assertTrue(any("Normal Operation" in n for n in names))
        self.assertTrue(any("Gradual Heating" in n for n in names))
        self.assertTrue(any("Rapid Temperature Rise" in n for n in names))
        self.assertTrue(any("Sudden Temperature Spike" in n for n in names))
        self.assertTrue(any("Overheating" in n for n in names))
        self.assertTrue(any("Sustained High" in n for n in names))
        self.assertTrue(any("Abnormal Cooling" in n for n in names))
        self.assertTrue(any("Critical Condition" in n for n in names))
        self.assertTrue(any("Recovery" in n for n in names))
        self.assertTrue(any("Sub-threshold Anomaly" in n for n in names))
        self.assertTrue(any("Temporary Noise" in n for n in names))
        self.assertTrue(any("Persistent Thermal Anomaly" in n for n in names))

    def test_scenario_runner_execution_and_report(self):
        """Verify ThermalScenarioRunner executes scenario through real pipeline and returns valid report."""
        runner = ThermalScenarioRunner()
        scenario = get_overheating_scenario()
        report = runner.run(scenario, seed=42)

        self.assertIsInstance(report, ThermalScenarioReport)
        self.assertEqual(report.scenario_name, scenario.name)
        self.assertEqual(report.sensor_id, scenario.sensor_id)
        self.assertGreater(report.total_samples, 0)
        self.assertIn("OVERHEATING", report.all_conditions_detected)
        self.assertEqual(report.final_health_status, "CRITICAL")
        self.assertTrue(report.passed)

    def test_scenario_report_to_dict(self):
        """Verify ThermalScenarioReport.to_dict produces clean serializable dictionary."""
        runner = ThermalScenarioRunner()
        scenario = get_normal_operation_scenario()
        report = runner.run(scenario, seed=42)

        rep_dict = report.to_dict()
        self.assertEqual(rep_dict["scenario_name"], scenario.name)
        self.assertEqual(rep_dict["final_health_status"], "HEALTHY")
        self.assertIsInstance(rep_dict["all_active_alerts"], list)
        self.assertTrue(rep_dict["passed"])


if __name__ == "__main__":
    unittest.main()
