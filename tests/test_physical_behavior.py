"""Physical behavior and sanity assertion test suite for ThermoShift simulation pipeline."""

import math
import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestPhysicalBehavior(unittest.TestCase):
    """Sanity test physical consistency across Workload -> Operating Conditions -> Power -> Temperature."""

    def setUp(self) -> None:
        """Initialize pipeline with physically realistic defaults."""
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
            thermal_resistance=1.5,  # 1.5 °C/W
            thermal_capacitance=20.0,  # tau = 30.0 s
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

    def test_higher_workload_yields_higher_power(self) -> None:
        """Requirement 1: Higher workload results in higher operating activity/power."""
        res_low = self.simulation.run(workload_profile=0.2, duration=2.0, timestep=1.0)
        res_high = self.simulation.run(workload_profile=0.9, duration=2.0, timestep=1.0)

        self.assertGreater(res_high.total_powers[0], res_low.total_powers[0])
        self.assertGreater(res_high.dynamic_powers[0], res_low.dynamic_powers[0])

    def test_higher_power_yields_higher_eventual_temperature(self) -> None:
        """Requirement 2: Sustained higher power produces higher eventual steady-state temperature."""
        res_low = self.simulation.run(workload_profile=0.3, duration=150.0, timestep=1.0)
        res_high = self.simulation.run(workload_profile=0.8, duration=150.0, timestep=1.0)

        self.assertGreater(res_high.final_temperature, res_low.final_temperature)

    def test_thermal_inertia_gradual_response(self) -> None:
        """Requirements 3 & 5: Temperature responds gradually due to thermal mass, not instantaneously."""
        # Instant step change in workload at t=0
        res = self.simulation.run(workload_profile=1.0, duration=10.0, timestep=1.0)

        # Power jumps instantly at t=0
        self.assertGreater(res.total_powers[0], 10.0)

        # Temperature at t=0 is initial temperature (25.0 °C)
        self.assertEqual(res.temperatures[0], 25.0)

        # Temperature at t=1.0 has increased, but is far from final steady state (~70+ °C)
        self.assertGreater(res.temperatures[1], 25.0)
        self.assertLess(res.temperatures[1], 40.0)

    def test_workload_drop_cooldown_behavior(self) -> None:
        """Requirements 4 & 7: Removing/reducing workload reduces power and cools toward new equilibrium."""
        # Step from 100% load for 50s to 10% load for 50s
        prof = WorkloadProfile.from_step([(0.0, 1.0), (50.0, 0.1)], interpolation="step")
        res = self.simulation.run(workload_profile=prof, duration=100.0, timestep=1.0)

        # Temp at t=50 (end of heavy load) should be high
        t_peak = res.temperatures[50]
        t_final = res.final_temperature

        # Power drops immediately at t=50
        self.assertGreater(res.total_powers[49], res.total_powers[51])

        # Temperature cools down gradually after t=50
        self.assertGreater(t_peak, t_final)
        self.assertGreater(t_final, 25.0)

    def test_sustained_workload_approaches_equilibrium(self) -> None:
        """Requirement 6: With sustained workload, temperature approaches thermal equilibrium."""
        res = self.simulation.run(workload_profile=0.6, duration=200.0, timestep=1.0)

        summary = res.summary()
        steady_state_est = summary.steady_state_final_estimate

        # Final temperature after 200s ( > 6 * tau ) should be very close to steady state
        self.assertAlmostEqual(res.final_temperature, steady_state_est, delta=0.5)

    def test_no_nan_or_inf_in_simulation_outputs(self) -> None:
        """Requirement 8: No NaN or infinite values occur for valid inputs."""
        prof = WorkloadProfile.pattern_mixed(duration=50.0)
        res = self.simulation.run(workload_profile=prof, duration=50.0, timestep=0.5)

        for val in res.times:
            self.assertTrue(math.isfinite(val))
        for val in res.workloads:
            self.assertTrue(math.isfinite(val))
        for val in res.total_powers:
            self.assertTrue(math.isfinite(val))
        for val in res.temperatures:
            self.assertTrue(math.isfinite(val))


if __name__ == "__main__":
    unittest.main()
