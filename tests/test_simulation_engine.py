"""Unit tests for ThermoShift simulation engine orchestration, validation, and integration calls."""

import unittest
from unittest.mock import MagicMock

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.power.power_result import PowerResult
from src.simulation.config import SimulationConfig
from src.simulation.engine import ThermoShiftSimulation
from src.simulation.exceptions import InvalidSimulationParameterError
from src.simulation.result import ThermoShiftSimulationResult
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestSimulationEngine(unittest.TestCase):
    """Test ThermoShiftSimulation engine setup, run execution, and validation."""

    def setUp(self) -> None:
        """Initialize standard simulation test fixtures."""
        self.power_config = PowerConfig(
            capacitance=1.5e-9,
            base_static_power=5.0,
            min_voltage=0.8,
            max_voltage=1.2,
            min_frequency=1.0e9,
            max_frequency=3.0e9,
        )
        self.power_estimator = PowerEstimator(config=self.power_config)

        self.thermal_config = ThermalConfig(
            ambient_temperature=25.0,
            initial_temperature=25.0,
            thermal_resistance=1.0,
            thermal_capacitance=10.0,
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

    def test_default_simulation_initialization(self) -> None:
        """Test instantiation of ThermoShiftSimulation with default parameters."""
        sim = ThermoShiftSimulation()
        self.assertIsNotNone(sim.power_estimator)
        self.assertIsNotNone(sim.thermal_model)
        self.assertIsNotNone(sim.dvfs_mapper)
        self.assertIsNotNone(sim.config)

    def test_run_constant_workload_simulation(self) -> None:
        """Test simple constant workload simulation execution."""
        results = self.simulation.run(workload_profile=0.5, duration=5.0, timestep=1.0)

        self.assertIsInstance(results, ThermoShiftSimulationResult)
        self.assertEqual(len(results), 6)  # t = 0, 1, 2, 3, 4, 5
        self.assertEqual(results.times, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(all(w == 0.5 for w in results.workloads))

        # Check temperature evolves upwards from 25 °C
        self.assertEqual(results.temperatures[0], 25.0)
        self.assertGreater(results.temperatures[-1], 25.0)

    def test_pipeline_calls_power_estimator_and_thermal_model(self) -> None:
        """Integration test verifying simulation calls PowerEstimator and ThermalModel directly."""
        mock_power_estimator = MagicMock(spec=PowerEstimator)
        mock_power_estimator.config = self.power_config

        # Return realistic dummy PowerResult
        mock_power_result = PowerResult(
            dynamic_power=4.0,
            static_power=5.0,
            total_power=9.0,
            workload=0.5,
            voltage=1.0,
            frequency=2.0e9,
            activity_factor=0.5,
            capacitance=1.5e-9,
            temperature=25.0,
        )
        mock_power_estimator.estimate.return_value = mock_power_result

        mock_thermal_model = MagicMock(spec=ThermalModel)
        mock_thermal_model.ambient_temperature = 25.0
        mock_thermal_model.initial_temperature = 25.0
        mock_thermal_model.thermal_resistance = 1.0
        mock_thermal_model.thermal_capacitance = 10.0
        mock_thermal_model.current_temperature = 25.0
        mock_thermal_model.step.return_value = 26.5

        sim = ThermoShiftSimulation(
            power_estimator=mock_power_estimator,
            thermal_model=mock_thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

        res = sim.run(workload_profile=0.5, duration=2.0, timestep=1.0)

        # Verify PowerEstimator.estimate was called at each timestep
        self.assertGreaterEqual(mock_power_estimator.estimate.call_count, 2)

        # Verify ThermalModel.step was called to advance temperature
        self.assertGreaterEqual(mock_thermal_model.step.call_count, 2)

    def test_invalid_duration_and_timestep_raise_errors(self) -> None:
        """Test input validation for invalid duration and timestep parameters."""
        # Negative duration
        with self.assertRaises(InvalidSimulationParameterError):
            self.simulation.run(workload_profile=0.5, duration=-5.0, timestep=1.0)

        # Zero timestep
        with self.assertRaises(InvalidSimulationParameterError):
            self.simulation.run(workload_profile=0.5, duration=5.0, timestep=0.0)

        # Timestep greater than duration
        with self.assertRaises(InvalidSimulationParameterError):
            self.simulation.run(workload_profile=0.5, duration=5.0, timestep=10.0)

    def test_invalid_workload_raises_error(self) -> None:
        """Test invalid workload profile input raises InvalidSimulationParameterError."""
        with self.assertRaises(InvalidSimulationParameterError):
            self.simulation.run(workload_profile=1.5, duration=5.0, timestep=1.0)

        with self.assertRaises(InvalidSimulationParameterError):
            self.simulation.run(workload_profile=-0.2, duration=5.0, timestep=1.0)


if __name__ == "__main__":
    unittest.main()
