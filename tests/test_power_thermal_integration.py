"""Integration tests for end-to-end Power Estimator -> Thermal Model pipeline."""

import unittest

from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.thermal.thermal_result import ThermalSimulationResult


class TestPowerThermalIntegration(unittest.TestCase):
    """Test full pipeline integration: Operating conditions -> PowerEstimator -> ThermalModel."""

    def setUp(self) -> None:
        """Initialize both Power Estimator and Thermal Model subsystems."""
        self.power_config = PowerConfig(
            capacitance=1.5e-9,  # 1.5 nF
            base_static_power=5.0,  # 5.0 W
            min_voltage=0.5,
            max_voltage=2.0,
            min_frequency=5.0e8,
            max_frequency=6.0e9,
            reference_temperature=25.0,
            temperature_coefficient=0.015,
            static_power_model="LINEAR",
        )
        self.power_estimator = PowerEstimator(config=self.power_config)

        self.thermal_config = ThermalConfig(
            ambient_temperature=25.0,
            initial_temperature=25.0,
            thermal_resistance=1.2,  # °C/W
            thermal_capacitance=20.0,  # J/°C
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)

    def test_pipeline_direct_power_result_consumption(self) -> None:
        """Test that ThermalModel can directly consume PowerResult objects from PowerEstimator."""
        # Estimate power at 80% load, 1.1 V, 3.0 GHz, 25 °C
        power_res = self.power_estimator.estimate(
            workload=0.80,
            voltage=1.10,
            frequency=3.0e9,
            temperature=25.0,
        )
        # Dynamic: 0.8 * 1.5e-9 * 1.21 * 3.0e9 = 4.356 W
        # Static: 5.0 W
        # Total: 9.356 W
        self.assertAlmostEqual(power_res.total_power, 9.356, places=4)

        # 1. Direct steady state calculation using PowerResult
        t_steady = self.thermal_model.steady_state_temperature(power_res)
        # Expected: 25.0 + 9.356 * 1.2 = 25.0 + 11.2272 = 36.2272 °C
        self.assertAlmostEqual(t_steady, 36.2272, places=4)

        # 2. Direct single step using PowerResult
        t_step = self.thermal_model.step(power=power_res, dt=1.0)
        self.assertGreater(t_step, 25.0)
        self.assertLess(t_step, t_steady)

        # 3. Direct transient simulation using PowerResult
        sim_res = self.thermal_model.simulate(power=power_res, duration=150.0, timestep=1.0)
        self.assertIsInstance(sim_res, ThermalSimulationResult)
        self.assertAlmostEqual(sim_res.final_temperature, t_steady, delta=0.05)

    def test_pipeline_dvfs_workload_transition(self) -> None:
        """Test full thermal response to dynamic workload and DVFS state changes."""
        # Stage 1: Idle (workload = 0%, V = 0.8V, f = 1.0 GHz) -> 10 seconds
        p_idle = self.power_estimator.estimate(workload=0.0, voltage=0.8, frequency=1.0e9, temperature=25.0)

        # Stage 2: Heavy compute (workload = 100%, V = 1.25V, f = 3.8 GHz) -> 40 seconds
        p_heavy = self.power_estimator.estimate(workload=1.0, voltage=1.25, frequency=3.8e9, temperature=50.0)

        # Stage 3: Throttle/Cooldown (workload = 30%, V = 0.9V, f = 1.8 GHz) -> 50 seconds
        p_cooldown = self.power_estimator.estimate(workload=0.3, voltage=0.9, frequency=1.8e9, temperature=40.0)

        profile = [p_idle] * 10 + [p_heavy] * 40 + [p_cooldown] * 50
        sim_res = self.thermal_model.simulate_profile(power_profile=profile, timestep=1.0)

        # Temperature at end of Stage 1 (idle) should be near ambient
        t_after_idle = sim_res.temperatures[10]
        # Temperature at end of Stage 2 (heavy compute) should reach peak
        t_after_heavy = sim_res.temperatures[50]
        # Temperature at end of Stage 3 (cooldown) should drop
        t_final = sim_res.final_temperature

        self.assertLess(t_after_idle, t_after_heavy)
        self.assertGreater(t_after_heavy, t_final)
        self.assertEqual(sim_res.max_temperature, max(sim_res.temperatures))


if __name__ == "__main__":
    unittest.main()
