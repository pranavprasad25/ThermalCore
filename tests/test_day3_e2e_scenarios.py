"""Day 3 Task 6: End-to-End Scenarios, Numerical Stability, and Edge Case Tests."""

import math
import unittest

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.power.exceptions import InvalidPowerParameterError
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.safety.status import ThermalStatus
from src.simulation.config import SimulationConfig
from src.simulation.engine import ThermoShiftSimulation
from src.simulation.exceptions import InvalidSimulationParameterError
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidThermalParameterError
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


class TestDay3EndToEndScenarios(unittest.TestCase):
    """Integration test suite evaluating 5 realistic end-to-end simulation scenarios."""

    def setUp(self) -> None:
        """Initialize standard power, thermal, and simulation fixtures."""
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
            thermal_resistance=1.5,  # 1.5 °C/W
            thermal_capacitance=10.0, # 10 J/°C (tau = 15s)
        )
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)

        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

        self.safety_config = ThermalSafetyConfig(
            warning_temperature=35.0,
            critical_temperature=45.0,
            maximum_temperature=55.0,
        )
        self.safety_monitor = ThermalSafetyMonitor(config=self.safety_config)

    # -------------------------------------------------------------------------
    # SCENARIO 1 — LOW WORKLOAD
    # -------------------------------------------------------------------------
    def test_scenario_1_low_workload(self) -> None:
        """SCENARIO 1: 0-10s at 20% workload.
        Expected: Low power, temperature remains within normal range, no overheating.
        """
        results = self.simulation.run(workload_profile=0.20, duration=10.0, timestep=1.0)
        analysis = results.analyze_safety(self.safety_monitor)

        # Power assertion
        self.assertLess(results.summary().average_power, 15.0)
        # Temperature assertion: stays below 35°C (warning limit)
        self.assertLess(results.summary().peak_temperature, 35.0)
        # Safety status assertion: stays NORMAL throughout
        self.assertFalse(analysis.has_violations)
        self.assertFalse(analysis.has_overheating)
        self.assertEqual(analysis.normal_timesteps, len(results))

    # -------------------------------------------------------------------------
    # SCENARIO 2 — HIGH SUSTAINED WORKLOAD
    # -------------------------------------------------------------------------
    def test_scenario_2_high_sustained_workload(self) -> None:
        """SCENARIO 2: 0-20s at 95% sustained workload.
        Expected: Increased power, temperature rises, thermal safety responds.
        """
        results = self.simulation.run(workload_profile=0.95, duration=20.0, timestep=1.0)
        analysis = results.analyze_safety(self.safety_monitor)

        # Power & temperature rise assertion
        self.assertGreater(results.summary().peak_total_power, 10.0)
        self.assertGreater(results.summary().peak_temperature, 35.0)

        # Temperature rises monotonically from initial ambient 25°C
        for i in range(1, len(results)):
            self.assertGreaterEqual(results.temperatures[i], results.temperatures[i - 1])

        # Safety monitor detects elevated states
        self.assertTrue(analysis.has_violations)
        self.assertIsNotNone(analysis.first_warning_time)

    # -------------------------------------------------------------------------
    # SCENARIO 3 — VARIABLE WORKLOAD
    # -------------------------------------------------------------------------
    def test_scenario_3_variable_workload(self) -> None:
        """SCENARIO 3: 20% -> 80% -> 40% -> 100% -> 30% workload timeline.
        Expected: Power follows workload/operating conditions, temperature responds with inertia, status changes.
        """
        timeline = [(0.0, 0.20), (5.0, 0.80), (10.0, 0.40), (15.0, 1.00), (20.0, 0.30)]
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

        results = self.simulation.run(workload_profile=profile, duration=25.0, timestep=1.0)
        analysis = results.analyze_safety(self.safety_monitor)

        # Power follows workload dynamically
        self.assertGreater(results.total_powers[6], results.total_powers[2])  # 80% vs 20%
        self.assertLess(results.total_powers[11], results.total_powers[6])   # 40% vs 80%
        self.assertGreater(results.total_powers[16], results.total_powers[11]) # 100% vs 40%

        # Thermal inertia: temperature at t=16s (just after 100% step) continues rising into t=17-20s
        self.assertGreater(results.temperatures[18], results.temperatures[16])

        # Status transitions occur
        self.assertGreater(len(analysis.transitions), 0)

    # -------------------------------------------------------------------------
    # SCENARIO 4 — THERMAL RECOVERY
    # -------------------------------------------------------------------------
    def test_scenario_4_thermal_recovery(self) -> None:
        """SCENARIO 4: Start with high workload (100%), then drop to low workload (10%).
        Expected: Power decreases, temperature gradually moves toward lower equilibrium, thermal status recovers.
        """
        timeline = [(0.0, 1.00), (15.0, 0.10)]
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

        results = self.simulation.run(workload_profile=profile, duration=40.0, timestep=1.0)
        analysis = results.analyze_safety(self.safety_monitor)

        # High temperature at step change (t=15s)
        temp_peak = results.temperatures[15]
        # Temperature decays gradually after t=15s (does not instantly jump)
        temp_after_step = results.temperatures[16]
        temp_final = results.final_temperature

        self.assertLess(temp_after_step, temp_peak)
        self.assertLess(temp_final, temp_after_step)
        # Power drops immediately at t=15s
        self.assertLess(results.total_powers[16], results.total_powers[14])

        # Safety status recovers (WARNING/CRITICAL cleared transitions present)
        cleared_events = [t for t in analysis.transitions if "CLEARED" in t.event_type.value]
        self.assertGreater(len(cleared_events), 0)

    # -------------------------------------------------------------------------
    # SCENARIO 5 — ZERO POWER
    # -------------------------------------------------------------------------
    def test_scenario_5_zero_power(self) -> None:
        """SCENARIO 5: Zero power input to thermal model (e.g. initial temp 50°C, zero power).
        Expected: Temperature approaches ambient (25°C) without invalid numerical behavior.
        """
        model = ThermalModel(ambient_temperature=25.0, initial_temperature=50.0, thermal_resistance=1.5, thermal_capacitance=10.0)
        sim_res = model.simulate(power=0.0, duration=60.0, timestep=1.0)

        # Exponential decay towards ambient
        self.assertAlmostEqual(sim_res.final_temperature, 25.0, delta=0.5)
        # Verify no NaN or Inf
        for t_val in sim_res.temperatures:
            self.assertTrue(math.isfinite(t_val))
            self.assertGreaterEqual(t_val, 25.0)


class TestDay3NumericalStability(unittest.TestCase):
    """Test suite verifying transient thermal simulation numerical stability across timesteps and durations."""

    def setUp(self) -> None:
        self.power_config = PowerConfig(
            capacitance=1.5e-9,
            base_static_power=5.0,
            min_voltage=0.8,
            max_voltage=1.25,
            min_frequency=1.0e9,
            max_frequency=3.5e9,
        )
        self.power_estimator = PowerEstimator(config=self.power_config)
        self.thermal_config = ThermalConfig(ambient_temperature=25.0, initial_temperature=25.0, thermal_resistance=1.2, thermal_capacitance=12.0)
        self.thermal_model = ThermalModel(config=self.thermal_config)
        self.dvfs_mapper = LinearDVFSMapper(power_config=self.power_config)
        self.simulation = ThermoShiftSimulation(
            power_estimator=self.power_estimator,
            thermal_model=self.thermal_model,
            dvfs_mapper=self.dvfs_mapper,
        )

    def test_small_timestep_stability(self) -> None:
        """Verify numerical stability with small timestep dt=0.001s."""
        res = self.simulation.run(workload_profile=0.5, duration=1.0, timestep=0.001)
        self.assertEqual(len(res.temperatures), 1001)
        for t_val in res.temperatures:
            self.assertTrue(math.isfinite(t_val))
            self.assertGreaterEqual(t_val, 25.0)

    def test_large_timestep_stability(self) -> None:
        """Verify numerical stability with large timestep dt=10.0s (EXACT analytical integration)."""
        res = self.simulation.run(workload_profile=0.5, duration=100.0, timestep=10.0)
        self.assertEqual(len(res.temperatures), 11)
        for t_val in res.temperatures:
            self.assertTrue(math.isfinite(t_val))
            self.assertGreaterEqual(t_val, 25.0)

    def test_long_simulation_stability(self) -> None:
        """Verify numerical stability over a long simulation (1000s duration)."""
        res = self.simulation.run(workload_profile=0.7, duration=1000.0, timestep=10.0)
        self.assertEqual(len(res.temperatures), 101)
        self.assertTrue(math.isfinite(res.final_temperature))
        # Temperature reaches equilibrium T_steady
        p_final = res.total_powers[-1]
        t_steady_expected = 25.0 + p_final * 1.2
        self.assertAlmostEqual(res.final_temperature, t_steady_expected, delta=0.1)

    def test_rapid_workload_oscillation_stability(self) -> None:
        """Verify numerical stability under rapid workload oscillation (step function every 1s)."""
        timeline = [(float(t), 1.0 if t % 2 == 0 else 0.0) for t in range(30)]
        profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")
        res = self.simulation.run(workload_profile=profile, duration=30.0, timestep=0.5)

        for t_val in res.temperatures:
            self.assertTrue(math.isfinite(t_val))
            self.assertGreaterEqual(t_val, 25.0)
            self.assertLess(t_val, 150.0)


class TestDay3BoundaryEdgeCases(unittest.TestCase):
    """Test suite covering thermal safety boundary precision, initial temperature offsets, and invalid parameters."""

    def setUp(self) -> None:
        self.safety_config = ThermalSafetyConfig(
            warning_temperature=70.0,
            critical_temperature=85.0,
            maximum_temperature=95.0,
        )
        self.monitor = ThermalSafetyMonitor(config=self.safety_config)

    def test_exact_threshold_boundary_classifications(self) -> None:
        """Test exact equality boundaries: T = warning, T = critical, T = maximum."""
        # 69.99 °C -> NORMAL
        self.assertEqual(self.monitor.classify_status(69.99), ThermalStatus.NORMAL)
        # 70.00 °C -> WARNING (inclusive bound)
        self.assertEqual(self.monitor.classify_status(70.00), ThermalStatus.WARNING)
        # 70.01 °C -> WARNING
        self.assertEqual(self.monitor.classify_status(70.01), ThermalStatus.WARNING)

        # 84.99 °C -> WARNING
        self.assertEqual(self.monitor.classify_status(84.99), ThermalStatus.WARNING)
        # 85.00 °C -> CRITICAL (inclusive bound)
        self.assertEqual(self.monitor.classify_status(85.00), ThermalStatus.CRITICAL)
        # 85.01 °C -> CRITICAL
        self.assertEqual(self.monitor.classify_status(85.01), ThermalStatus.CRITICAL)

        # 94.99 °C -> CRITICAL
        self.assertEqual(self.monitor.classify_status(94.99), ThermalStatus.CRITICAL)
        # 95.00 °C -> OVERHEATING (inclusive bound)
        self.assertEqual(self.monitor.classify_status(95.00), ThermalStatus.OVERHEATING)
        # 95.01 °C -> OVERHEATING
        self.assertEqual(self.monitor.classify_status(95.01), ThermalStatus.OVERHEATING)

    def test_initial_temperature_above_and_below_ambient(self) -> None:
        """Test transient thermal model when initial temperature is above or below ambient."""
        # Initial temp above ambient (60°C > 25°C) with zero power -> cools down
        model_high = ThermalModel(ambient_temperature=25.0, initial_temperature=60.0, thermal_resistance=1.2, thermal_capacitance=10.0)
        res_high = model_high.simulate(power=0.0, duration=30.0, timestep=1.0)
        self.assertLess(res_high.final_temperature, 60.0)
        self.assertGreaterEqual(res_high.final_temperature, 25.0)

        # Initial temp below ambient (10°C < 25°C) with zero power -> warms up to ambient
        model_low = ThermalModel(ambient_temperature=25.0, initial_temperature=10.0, thermal_resistance=1.2, thermal_capacitance=10.0)
        res_low = model_low.simulate(power=0.0, duration=30.0, timestep=1.0)
        self.assertGreater(res_low.final_temperature, 10.0)
        self.assertLessEqual(res_low.final_temperature, 25.0)

    def test_invalid_parameter_rejections(self) -> None:
        """Test that invalid parameters raise descriptive exceptions across all Day 3 components."""
        # Invalid workload in PowerEstimator
        pe = PowerEstimator()
        with self.assertRaises(InvalidPowerParameterError):
            pe.estimate_dynamic_power(workload=-0.1, voltage=1.0, frequency=2.0e9)
        with self.assertRaises(InvalidPowerParameterError):
            pe.estimate_dynamic_power(workload=1.5, voltage=1.0, frequency=2.0e9)

        # Invalid voltage / frequency in PowerEstimator
        with self.assertRaises(InvalidPowerParameterError):
            pe.estimate_dynamic_power(workload=0.5, voltage=0.0, frequency=2.0e9)
        with self.assertRaises(InvalidPowerParameterError):
            pe.estimate_dynamic_power(workload=0.5, voltage=1.0, frequency=-1.0)

        # Invalid thermal model parameters
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_resistance=0.0)
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_capacitance=-5.0)

        # Invalid simulation duration / timestep
        sim = ThermoShiftSimulation()
        with self.assertRaises(InvalidSimulationParameterError):
            sim.run(workload_profile=0.5, duration=0.0)
        with self.assertRaises(InvalidSimulationParameterError):
            sim.run(workload_profile=0.5, duration=10.0, timestep=0.0)
        with self.assertRaises(InvalidSimulationParameterError):
            sim.run(workload_profile=0.5, duration=5.0, timestep=10.0)


if __name__ == "__main__":
    unittest.main()
