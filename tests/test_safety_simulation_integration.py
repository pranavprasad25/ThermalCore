"""Integration tests for Thermal Safety analysis with Day 3 Task 3 simulation results."""

import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.safety.status import ThermalStatus, ThermalTransitionEventType
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestSafetySimulationIntegration(unittest.TestCase):
    """Integration test suite connecting ThermoShiftSimulation results to ThermalSafetyMonitor."""

    def setUp(self) -> None:
        """Initialize Power, Thermal, Simulation, and Safety subsystems."""
        self.power_config = PowerConfig(
            capacitance=1.5e-9,
            base_static_power=5.0,
            min_voltage=0.8,
            max_voltage=1.25,
            min_frequency=1.0e9,
            max_frequency=3.5e9,
        )
        self.power_estimator = PowerEstimator(config=self.power_config)

        self.thermal_config = ThermalConfig(
            ambient_temperature=25.0,
            initial_temperature=25.0,
            thermal_resistance=1.5,  # 1.5 °C/W -> 15W total power reaches ~47.5°C; 50W reaches 100°C
            thermal_capacitance=10.0,
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

        self.safety_config = ThermalSafetyConfig(
            warning_temperature=35.0,   # Adjusted for test thermal limits
            critical_temperature=45.0,
            maximum_temperature=55.0,
        )
        self.safety_monitor = ThermalSafetyMonitor(config=self.safety_config)

    def test_simulation_analysis_no_violations(self) -> None:
        """Test simulation safety analysis when system remains in NORMAL state throughout."""
        # Low 10% constant workload -> low power -> temp stays ~28 °C (< 35 °C warning)
        res = self.simulation.run(workload_profile=0.1, duration=10.0, timestep=1.0)
        analysis = self.safety_monitor.analyze(res)

        self.assertFalse(analysis.has_violations)
        self.assertFalse(analysis.has_overheating)
        self.assertEqual(analysis.normal_timesteps, len(res))
        self.assertEqual(analysis.warning_timesteps, 0)
        self.assertEqual(analysis.critical_timesteps, 0)
        self.assertEqual(analysis.overheating_timesteps, 0)
        self.assertIsNone(analysis.first_warning_time)

    def test_simulation_analysis_with_transitions(self) -> None:
        """Test step workload causing transitions: NORMAL -> WARNING -> CRITICAL -> WARNING -> NORMAL."""
        # Step workload timeline: 0s->10%, 5s->60%, 15s->100%, 25s->10%
        timeline = [(0.0, 0.1), (5.0, 0.6), (15.0, 1.0), (25.0, 0.1)]
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

        res = self.simulation.run(workload_profile=profile, duration=40.0, timestep=1.0)

        # Use helper method on result directly
        analysis = res.analyze_safety(self.safety_monitor)

        self.assertTrue(analysis.has_violations)
        self.assertGreater(len(analysis.transitions), 0)

        # Verify first warning time recorded
        self.assertIsNotNone(analysis.first_warning_time)
        self.assertGreaterEqual(analysis.first_warning_time, 5.0)

        # Check transition events list
        event_types = [t.event_type for t in analysis.transitions]
        self.assertIn(ThermalTransitionEventType.WARNING_ENTERED, event_types)

    def test_simulation_analysis_overheating_and_recovery(self) -> None:
        """Test simulation where overheating threshold is crossed and then recovers."""
        # Heavy workload step causing temp to rise above 55 °C
        # Power config with high capacitance to generate > 20 W power
        high_power_cfg = PowerConfig(capacitance=3.0e-9, base_static_power=10.0, max_voltage=1.4, max_frequency=4.0e9)
        high_power_est = PowerEstimator(config=high_power_cfg)
        high_dvfs = LinearDVFSMapper(power_config=high_power_cfg)

        sim = ThermoShiftSimulation(power_estimator=high_power_est, thermal_model=self.thermal_model, dvfs_mapper=high_dvfs)

        timeline = [(0.0, 1.0), (20.0, 0.0)]
        prof = WorkloadProfile.from_step(steps=timeline, interpolation="step")

        res = sim.run(workload_profile=prof, duration=40.0, timestep=1.0)
        analysis = self.safety_monitor.analyze(res)

        self.assertTrue(analysis.has_overheating)
        self.assertIsNotNone(analysis.first_overheating_time)
        self.assertGreater(analysis.overheating_timesteps, 0)

        # Verify de-escalation / recovery transitions
        event_types = [t.event_type for t in analysis.transitions]
        self.assertIn(ThermalTransitionEventType.OVERHEATING_ENTERED, event_types)
        self.assertIn(ThermalTransitionEventType.OVERHEATING_CLEARED, event_types)


if __name__ == "__main__":
    unittest.main()
