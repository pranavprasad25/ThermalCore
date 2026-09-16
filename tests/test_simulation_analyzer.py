"""Unit tests for SimulationAnalyzer and SimulationAnalysisSummary."""

import math
import unittest

from src.analysis.analyzer import SimulationAnalyzer
from src.analysis.exceptions import InvalidAnalysisError
from src.analysis.summary import SimulationAnalysisSummary
from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.simulation.engine import ThermoShiftSimulation
from src.simulation.result import ThermoShiftSimulationResult
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestSimulationAnalyzer(unittest.TestCase):
    """Test suite for SimulationAnalyzer statistical analysis and safety metrics."""

    def setUp(self) -> None:
        """Set up standard simulation engine and run fixture simulation."""
        self.power_config = PowerConfig(capacitance=1.5e-9, base_static_power=5.0)
        self.power_estimator = PowerEstimator(config=self.power_config)
        self.thermal_config = ThermalConfig(ambient_temperature=25.0, initial_temperature=25.0, thermal_resistance=1.5)
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

        timeline = [(0.0, 0.2), (5.0, 0.8), (15.0, 0.4)]
        self.profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")
        self.results = self.simulation.run(workload_profile=self.profile, duration=25.0, timestep=1.0)

        self.analyzer = SimulationAnalyzer()
        self.safety_config = ThermalSafetyConfig(warning_temperature=35.0, critical_temperature=45.0, maximum_temperature=55.0)
        self.safety_monitor = ThermalSafetyMonitor(config=self.safety_config)

    def test_analyzer_returns_summary(self) -> None:
        """Test that analyzer returns a valid SimulationAnalysisSummary object."""
        summary = self.analyzer.analyze(self.results, safety_monitor=self.safety_monitor)
        self.assertIsInstance(summary, SimulationAnalysisSummary)

        # Workload metrics
        self.assertEqual(summary.peak_workload, 0.8)
        self.assertAlmostEqual(summary.min_workload, 0.2)
        self.assertGreater(summary.average_workload, 0.2)
        self.assertLess(summary.average_workload, 0.8)

        # Power metrics
        self.assertGreater(summary.peak_total_power, summary.min_total_power)
        self.assertGreater(summary.average_total_power, summary.min_total_power)

        # Temperature metrics
        self.assertEqual(summary.final_temperature, self.results.temperatures[-1])
        self.assertGreaterEqual(summary.peak_temperature, summary.final_temperature)

        # Output formatting
        d = summary.to_dict()
        self.assertIn("workload", d)
        self.assertIn("power", d)
        self.assertIn("temperature", d)
        self.assertIn("thermal_safety", d)

        s_str = summary.to_string()
        self.assertIn("ThermoShift Simulation Analysis Summary", s_str)

    def test_analyzer_input_validation(self) -> None:
        """Test analyzer validation rejects non-result, empty, or corrupted data."""
        with self.assertRaises(InvalidAnalysisError):
            self.analyzer.analyze("invalid_input_type")  # type: ignore

        # Corrupted array lengths
        corrupted_results = ThermoShiftSimulationResult(
            times=[0.0, 1.0],
            workloads=[0.5],  # mismatch length
            frequencies=[2.0e9, 2.0e9],
            voltages=[1.0, 1.0],
            dynamic_powers=[5.0, 5.0],
            static_powers=[5.0, 5.0],
            total_powers=[10.0, 10.0],
            temperatures=[25.0, 26.0],
            steps=self.results.steps[:2],
            config=self.results.config,
            power_config=self.power_config,
            thermal_config=self.thermal_config,
        )
        with self.assertRaises(InvalidAnalysisError):
            self.analyzer.analyze(corrupted_results)

    def test_analyzer_non_monotonic_timestamps_rejection(self) -> None:
        """Test analyzer rejects non-monotonic timestamp sequences."""
        corrupted_results = ThermoShiftSimulationResult(
            times=[0.0, 2.0, 1.0],  # out of order
            workloads=[0.5, 0.5, 0.5],
            frequencies=[2.0e9, 2.0e9, 2.0e9],
            voltages=[1.0, 1.0, 1.0],
            dynamic_powers=[5.0, 5.0, 5.0],
            static_powers=[5.0, 5.0, 5.0],
            total_powers=[10.0, 10.0, 10.0],
            temperatures=[25.0, 26.0, 27.0],
            steps=self.results.steps[:3],
            config=self.results.config,
            power_config=self.power_config,
            thermal_config=self.thermal_config,
        )
        with self.assertRaises(InvalidAnalysisError):
            self.analyzer.analyze(corrupted_results)


if __name__ == "__main__":
    unittest.main()
