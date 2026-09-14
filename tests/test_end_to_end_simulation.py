"""Realistic End-to-End Integration Test for ThermoShift pipeline."""

import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestEndToEndSimulationIntegration(unittest.TestCase):
    """Integration test verifying complete Workload -> Operating Conditions -> Power -> Thermal model flow."""

    def test_realistic_workload_timeline(self) -> None:
        """Integration test with realistic workload profile:

        0s -> 20%
        2s -> 50%
        4s -> 90%
        6s -> 40%
        8s -> 80%
        """
        power_config = PowerConfig(
            capacitance=1.5e-9,
            base_static_power=5.0,
            min_voltage=0.8,
            max_voltage=1.2,
            min_frequency=1.0e9,
            max_frequency=3.0e9,
        )
        power_estimator = PowerEstimator(config=power_config)

        thermal_config = ThermalConfig(
            ambient_temperature=25.0,
            initial_temperature=25.0,
            thermal_resistance=1.2,
            thermal_capacitance=10.0,
        )
        thermal_model = ThermalModel(config=thermal_config)
        dvfs_mapper = LinearDVFSMapper(power_config=power_config)

        simulation = ThermoShiftSimulation(
            power_estimator=power_estimator,
            thermal_model=thermal_model,
            dvfs_mapper=dvfs_mapper,
        )

        timeline = [
            (0.0, 0.20),
            (2.0, 0.50),
            (4.0, 0.90),
            (6.0, 0.40),
            (8.0, 0.80),
        ]
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

        results = simulation.run(workload_profile=profile, duration=10.0, timestep=1.0)

        # 1. Verify result timestamps
        self.assertEqual(results.times, [float(i) for i in range(11)])

        # 2. Verify operating conditions derived at each step
        for k in range(len(results)):
            w = results.workloads[k]
            v = results.voltages[k]
            f = results.frequencies[k]

            # Verify linear DVFS mapping: V = 0.8 + w * 0.4, f = 1e9 + w * 2e9
            expected_v = 0.8 + w * 0.4
            expected_f = 1.0e9 + w * 2.0e9

            self.assertAlmostEqual(v, expected_v, places=4)
            self.assertAlmostEqual(f, expected_f, places=4)

        # 3. Verify PowerEstimator output matches expectation derived from physics equations
        for k in range(len(results)):
            w = results.workloads[k]
            v = results.voltages[k]
            f = results.frequencies[k]
            t = results.temperatures[k]

            # Direct PowerEstimator check
            expected_power = power_estimator.estimate(workload=w, voltage=v, frequency=f, temperature=t)

            self.assertAlmostEqual(results.dynamic_powers[k], expected_power.dynamic_power, places=4)
            self.assertAlmostEqual(results.static_powers[k], expected_power.static_power, places=4)
            self.assertAlmostEqual(results.total_powers[k], expected_power.total_power, places=4)

        # 4. Verify thermal transient consistency across high/low power phases
        # High power at t=4s (90% load) should be higher than t=0s (20% load)
        self.assertGreater(results.total_powers[4], results.total_powers[0])

        # Temperature at t=5s (after 90% load) should be rising rapidly
        self.assertGreater(results.temperatures[5], results.temperatures[4])

        # Power drops at t=6s (40% load)
        self.assertLess(results.total_powers[6], results.total_powers[5])

        # Summary assertions
        summary = results.summary()
        self.assertEqual(summary.peak_workload, 0.90)
        self.assertGreater(summary.peak_temperature, 25.0)
        self.assertEqual(summary.total_steps, 11)


if __name__ == "__main__":
    unittest.main()
