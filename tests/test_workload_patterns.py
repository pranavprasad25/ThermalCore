"""Unit and integration tests for different workload patterns in the simulation pipeline."""

import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestWorkloadPatterns(unittest.TestCase):
    """Test ThermoShift simulation under distinct workload patterns A, B, C, D, E."""

    def setUp(self) -> None:
        """Initialize Power Estimator, Thermal Model, and Simulation Engine."""
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
            thermal_resistance=1.2,
            thermal_capacitance=15.0,  # tau = 18.0 s
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

    def test_pattern_a_constant_workload(self) -> None:
        """Pattern A: Constant 80% workload for entire simulation."""
        prof = WorkloadProfile.pattern_constant(workload=0.8, duration=20.0)
        res = self.simulation.run(workload_profile=prof, duration=20.0, timestep=1.0)

        # Workload is constant 0.8
        self.assertTrue(all(w == 0.8 for w in res.workloads))
        # Total power is constant (or slightly increasing if temp-dependent static power)
        self.assertAlmostEqual(res.total_powers[0], res.total_powers[-1], delta=0.5)

        # Temperature smoothly rises toward steady state
        for i in range(len(res.temperatures) - 1):
            self.assertLessEqual(res.temperatures[i], res.temperatures[i + 1])

        self.assertGreater(res.final_temperature, 25.0)

    def test_pattern_b_step_workload(self) -> None:
        """Pattern B: Step workload (20% -> 80% -> 20%)."""
        prof = WorkloadProfile.pattern_step(
            initial_workload=0.2,
            high_workload=0.8,
            final_workload=0.2,
            step_times=(10.0, 40.0),
            duration=70.0,
        )
        res = self.simulation.run(workload_profile=prof, duration=70.0, timestep=1.0)

        # Step 1: Low workload (0.2) -> low power
        p_low = res.total_powers[5]

        # Step 2: High workload (0.8) -> high power
        p_high = res.total_powers[25]

        # Step 3: Cooldown workload (0.2) -> low power
        p_mid = res.total_powers[60]

        self.assertLess(p_low, p_high)
        self.assertGreater(p_high, p_mid)

        # Temperature during step 2 should reach peak and cool down in step 3
        t_high = max(res.temperatures)
        t_final = res.final_temperature
        self.assertGreater(t_high, t_final)

    def test_pattern_c_increasing_workload(self) -> None:
        """Pattern C: Increasing workload (20% -> 40% -> 60% -> 80% -> 100%)."""
        prof = WorkloadProfile.pattern_increasing(start=0.2, end=1.0, num_steps=5, duration=20.0)
        res = self.simulation.run(workload_profile=prof, duration=20.0, timestep=1.0)

        # Power generally increases across simulation
        self.assertLess(res.total_powers[0], res.total_powers[-1])
        # Peak temperature occurs at or near end
        summary = res.summary()
        self.assertEqual(summary.peak_workload, 1.0)
        self.assertGreaterEqual(summary.time_at_peak_temperature, 15.0)

    def test_pattern_d_decreasing_workload(self) -> None:
        """Pattern D: Decreasing workload (100% -> 80% -> 50% -> 20%)."""
        prof = WorkloadProfile.pattern_decreasing(start=1.0, end=0.2, num_steps=5, duration=50.0)
        res = self.simulation.run(workload_profile=prof, duration=50.0, timestep=1.0, initial_temperature=40.0)

        # Initial power should be much higher than final power
        self.assertGreater(res.total_powers[0], res.total_powers[-1])
        # Temperature starts high and cools down as workload decreases
        self.assertGreater(max(res.temperatures), res.final_temperature)

    def test_pattern_e_mixed_workload(self) -> None:
        """Pattern E: Mixed workload (20% -> 90% -> 40% -> 100% -> 30%)."""
        prof = WorkloadProfile.pattern_mixed(duration=25.0)
        res = self.simulation.run(workload_profile=prof, duration=25.0, timestep=1.0)

        # Verify dynamic responses
        self.assertEqual(len(res), 26)
        summary = res.summary()
        self.assertAlmostEqual(summary.peak_workload, 1.0, delta=0.01)
        self.assertGreater(summary.peak_temperature, 25.0)
        self.assertGreater(summary.peak_total_power, summary.average_power)


if __name__ == "__main__":
    unittest.main()
