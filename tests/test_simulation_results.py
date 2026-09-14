"""Unit tests for ThermoShift simulation result objects and summary metrics."""

import unittest

from src.simulation.engine import ThermoShiftSimulation
from src.simulation.result import SimulationSummary, ThermoShiftSimulationResult


class TestSimulationResults(unittest.TestCase):
    """Test ThermoShiftSimulationResult methods, iteration, indexing, and summary metrics."""

    def setUp(self) -> None:
        """Run a standard simulation to produce a test result instance."""
        sim = ThermoShiftSimulation()
        self.result = sim.run(workload_profile=0.5, duration=10.0, timestep=1.0)

    def test_result_structure_and_length(self) -> None:
        """Test result length and element types."""
        self.assertEqual(len(self.result), 11)  # t = 0 to 10
        self.assertEqual(len(self.result.times), 11)
        self.assertEqual(len(self.result.temperatures), 11)
        self.assertEqual(len(self.result.total_powers), 11)

        step0 = self.result[0]
        self.assertEqual(step0.step_index, 0)
        self.assertEqual(step0.time, 0.0)
        self.assertEqual(step0.workload, 0.5)

    def test_result_iteration(self) -> None:
        """Test iteration over simulation steps."""
        steps_list = list(self.result)
        self.assertEqual(len(steps_list), 11)
        for i, step in enumerate(steps_list):
            self.assertEqual(step.step_index, i)

    def test_get_variable(self) -> None:
        """Test get_variable by name."""
        times = self.result.get_variable("time")
        self.assertEqual(times, self.result.times)

        temps = self.result.get_variable("temperature")
        self.assertEqual(temps, self.result.temperatures)

        powers = self.result.get_variable("total_power")
        self.assertEqual(powers, self.result.total_powers)

        with self.assertRaises(KeyError):
            self.result.get_variable("invalid_var_name")

    def test_summary_metrics(self) -> None:
        """Test calculated summary metrics."""
        summary = self.result.summary()
        self.assertIsInstance(summary, SimulationSummary)
        self.assertEqual(summary.peak_workload, 0.5)
        self.assertGreater(summary.peak_temperature, 25.0)
        self.assertGreater(summary.average_power, 0.0)
        self.assertEqual(summary.total_steps, 11)

        summary_dict = summary.to_dict()
        self.assertIn("peak_temperature", summary_dict)
        self.assertIn("peak_total_power", summary_dict)

    def test_to_dict_export(self) -> None:
        """Test full ThermoShiftSimulationResult serialization to dictionary."""
        d = self.result.to_dict()
        self.assertIn("summary", d)
        self.assertIn("config", d)
        self.assertIn("steps", d)
        self.assertEqual(len(d["steps"]), 11)


if __name__ == "__main__":
    unittest.main()
