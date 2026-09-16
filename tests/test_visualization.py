"""Unit tests for ThermoShift Visualization functions and SimulationPlotter."""

import os
import tempfile
import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.visualization.exceptions import InvalidVisualizationError
from src.visualization.plotter import (
    SimulationPlotter,
    plot_overview,
    plot_power,
    plot_temperature,
    plot_workload,
)
from src.workload.profile import WorkloadProfile


class TestVisualization(unittest.TestCase):
    """Test suite for plotting functions, return types, threshold overlays, and exports."""

    def setUp(self) -> None:
        """Initialize simulation fixtures."""
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
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")
        self.results = self.simulation.run(workload_profile=profile, duration=20.0, timestep=1.0)

        self.safety_config = ThermalSafetyConfig(warning_temperature=35.0, critical_temperature=45.0, maximum_temperature=55.0)
        self.plotter = SimulationPlotter(safety_config=self.safety_config)

    def test_plot_workload_return_types(self) -> None:
        """Test plot_workload returns Figure and Axes objects."""
        fig, ax = plot_workload(self.results, show=False)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), "Time (s)")

    def test_plot_power_return_types(self) -> None:
        """Test plot_power returns Figure and Axes objects."""
        fig, ax = plot_power(self.results, show=False)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_ylabel(), "Power (W)")

    def test_plot_temperature_return_types(self) -> None:
        """Test plot_temperature returns Figure and Axes objects with threshold overlays."""
        fig, ax = plot_temperature(
            self.results,
            safety_config=self.safety_config,
            show_thresholds=True,
            show_violations=True,
            show=False,
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_ylabel(), "Temperature (°C)")

    def test_plot_overview_return_types(self) -> None:
        """Test plot_overview returns Figure and tuple of 3 Axes objects."""
        fig, (ax1, ax2, ax3) = plot_overview(
            self.results,
            safety_config=self.safety_config,
            show=False,
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax1)
        self.assertIsNotNone(ax2)
        self.assertIsNotNone(ax3)

    def test_result_shortcut_plotting_methods(self) -> None:
        """Test ThermoShiftSimulationResult shortcut plot methods."""
        fig1, ax1 = self.results.plot_workload(show=False)
        fig2, ax2 = self.results.plot_power(show=False)
        fig3, ax3 = self.results.plot_temperature(show=False)
        fig4, (a, b, c) = self.results.plot_overview(show=False)

        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)
        self.assertIsNotNone(fig4)

    def test_plot_file_export(self) -> None:
        """Test saving plot figures to PNG file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_plot.png")
            fig, ax = plot_workload(self.results, save_path=file_path, show=False)
            self.assertTrue(os.path.exists(file_path))
            self.assertGreater(os.path.getsize(file_path), 0)

    def test_invalid_result_plotting_raises_error(self) -> None:
        """Test plotting functions reject invalid results."""
        with self.assertRaises(InvalidVisualizationError):
            plot_workload("invalid_result", show=False)  # type: ignore


if __name__ == "__main__":
    unittest.main()
