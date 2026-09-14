"""Automated unit test suite for Day 3 — Task 2: Thermal RC Model."""

import math
import unittest

from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.power.power_result import PowerResult
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import (
    InvalidThermalParameterError,
    ThermalModelError,
    ThermalSimulationError,
)
from src.thermal.thermal_model import ThermalModel
from src.thermal.thermal_result import ThermalSimulationResult, ThermalStepResult


class TestThermalModel(unittest.TestCase):
    """Unit tests for ThermalModel covering steady-state, transient dynamics, validation, and integration."""

    def setUp(self) -> None:
        """Set up standard thermal model test fixture."""
        self.r_th = 1.2  # 1.2 °C/W
        self.c_th = 20.0  # 20.0 J/°C
        self.t_amb = 25.0  # 25.0 °C
        self.t_init = 25.0  # 25.0 °C
        self.model = ThermalModel(
            thermal_resistance=self.r_th,
            thermal_capacitance=self.c_th,
            ambient_temperature=self.t_amb,
            initial_temperature=self.t_init,
        )

    # --------------------------------------------------------------------------
    # 1. Steady-State Calculations
    # --------------------------------------------------------------------------

    def test_steady_state_temperature_exact(self) -> None:
        """Test steady state formula: T_steady = T_ambient + P * R_th."""
        # P = 25.0 W, R_th = 1.2 °C/W, T_amb = 25.0 °C
        # T_steady = 25.0 + (25.0 * 1.2) = 25.0 + 30.0 = 55.0 °C
        t_steady = self.model.steady_state_temperature(power=25.0)
        self.assertAlmostEqual(t_steady, 55.0, places=6)

    def test_temperature_rise_calculation(self) -> None:
        """Test temperature rise formula: Delta_T = P * R_th."""
        delta_t = self.model.temperature_rise(power=20.0)
        # Delta_T = 20.0 * 1.2 = 24.0 °C
        self.assertAlmostEqual(delta_t, 24.0, places=6)

    def test_steady_state_zero_power(self) -> None:
        """Test that with zero power (P = 0), steady-state temperature equals ambient temperature."""
        t_steady = self.model.steady_state_temperature(power=0.0)
        self.assertEqual(t_steady, self.t_amb)

    def test_steady_state_with_ambient_override(self) -> None:
        """Test steady-state temperature with overridden ambient temperature."""
        t_steady = self.model.steady_state_temperature(power=10.0, ambient_temperature=35.0)
        # T_steady = 35.0 + 10.0 * 1.2 = 47.0 °C
        self.assertAlmostEqual(t_steady, 47.0, places=6)

    def test_steady_state_with_r_th_override(self) -> None:
        """Test steady-state temperature with overridden thermal resistance."""
        t_steady = self.model.steady_state_temperature(power=10.0, thermal_resistance=2.0)
        # T_steady = 25.0 + 10.0 * 2.0 = 45.0 °C
        self.assertAlmostEqual(t_steady, 45.0, places=6)

    # --------------------------------------------------------------------------
    # 2. Time Constant & Properties
    # --------------------------------------------------------------------------

    def test_time_constant_tau(self) -> None:
        """Test that thermal time constant tau = R_th * C_th."""
        # tau = 1.2 * 20.0 = 24.0 seconds
        self.assertAlmostEqual(self.model.time_constant, 24.0, places=6)
        self.assertAlmostEqual(self.model.tau, 24.0, places=6)

    def test_initial_properties(self) -> None:
        """Test property getters on ThermalModel."""
        self.assertEqual(self.model.thermal_resistance, 1.2)
        self.assertEqual(self.model.thermal_capacitance, 20.0)
        self.assertEqual(self.model.ambient_temperature, 25.0)
        self.assertEqual(self.model.initial_temperature, 25.0)
        self.assertEqual(self.model.current_temperature, 25.0)

    # --------------------------------------------------------------------------
    # 3. Transient Heating Dynamics
    # --------------------------------------------------------------------------

    def test_transient_constant_power_convergence(self) -> None:
        """Test that transient simulation under constant power converges to T_steady."""
        power = 30.0  # W -> T_steady = 25.0 + 30.0 * 1.2 = 61.0 °C
        duration = 150.0  # > 5 * tau (5 * 24 = 120s)
        timestep = 0.5

        result = self.model.simulate(power=power, duration=duration, timestep=timestep)

        self.assertIsInstance(result, ThermalSimulationResult)
        self.assertAlmostEqual(result.times[0], 0.0)
        self.assertAlmostEqual(result.temperatures[0], 25.0)

        # Monotonically increasing during heating
        for i in range(len(result.temperatures) - 1):
            self.assertLessEqual(result.temperatures[i], result.temperatures[i + 1])

        # Final temperature should be within 0.5% of steady state (61.0 °C)
        self.assertAlmostEqual(result.final_temperature, 61.0, delta=0.1)
        self.assertAlmostEqual(result.steady_state_temperature, 61.0, places=4)

    def test_transient_tau_exponential_fraction(self) -> None:
        """Test physical property: after 1 tau (24s), temperature rises by ~63.2% of Delta_T."""
        power = 20.0  # Delta_T = 24.0 °C, T_steady = 49.0 °C
        tau = 24.0
        result = self.model.simulate(power=power, duration=tau, timestep=0.1)

        # Theoretical temperature at t = tau: T_amb + Delta_T * (1 - 1/e) = 25.0 + 24.0 * (1 - e^-1)
        expected_t_at_tau = 25.0 + (24.0 * (1.0 - math.exp(-1.0)))  # ~40.17 °C
        self.assertAlmostEqual(result.final_temperature, expected_t_at_tau, places=2)

    # --------------------------------------------------------------------------
    # 4. Transient Cooling Dynamics
    # --------------------------------------------------------------------------

    def test_cooling_from_elevated_initial_temperature(self) -> None:
        """Test that a hot processor cools toward ambient temperature when zero power is applied."""
        hot_model = ThermalModel(
            thermal_resistance=1.2,
            thermal_capacitance=20.0,
            ambient_temperature=25.0,
            initial_temperature=85.0,  # Hot start
        )

        result = hot_model.simulate(power=0.0, duration=180.0, timestep=0.5)

        # Verify monotonic decrease
        for i in range(len(result.temperatures) - 1):
            self.assertGreaterEqual(result.temperatures[i], result.temperatures[i + 1])

        # Final temperature converges to ambient (25.0 °C)
        self.assertAlmostEqual(result.final_temperature, 25.0, delta=0.05)

    # --------------------------------------------------------------------------
    # 5. Discrete Step Integration & Reset
    # --------------------------------------------------------------------------

    def test_step_method_stateful_advancement(self) -> None:
        """Test single-step temperature evolution using model.step()."""
        self.model.reset(initial_temperature=25.0)
        self.assertEqual(self.model.current_temperature, 25.0)

        # Step 1 second with 20 W
        t1 = self.model.step(power=20.0, dt=1.0)
        self.assertGreater(t1, 25.0)
        self.assertEqual(self.model.current_temperature, t1)

        # Step another 1 second
        t2 = self.model.step(power=20.0, dt=1.0)
        self.assertGreater(t2, t1)

        # Reset
        self.model.reset(initial_temperature=30.0)
        self.assertEqual(self.model.current_temperature, 30.0)

    # --------------------------------------------------------------------------
    # 6. Time-Varying Power & Profiles
    # --------------------------------------------------------------------------

    def test_time_varying_callable_power(self) -> None:
        """Test simulation with a callable time-varying power function."""
        # Square pulse: 40 W for first 20s, then 5 W
        def pulse_power(t: float) -> float:
            return 40.0 if t < 20.0 else 5.0

        result = self.model.simulate(power=pulse_power, duration=60.0, timestep=0.5)
        self.assertEqual(len(result.times), len(result.temperatures))

        # Peak temperature should occur around t = 20s
        idx_20s = int(20.0 / 0.5)
        t_at_pulse_end = result.temperatures[idx_20s]
        t_final = result.final_temperature

        # Temperature at pulse end should be significantly higher than final temperature (cooled down)
        self.assertGreater(t_at_pulse_end, t_final)

    def test_simulate_profile_sequence(self) -> None:
        """Test simulate_profile with discrete power array."""
        profile = [10.0] * 10 + [50.0] * 20 + [0.0] * 30
        result = self.model.simulate_profile(power_profile=profile, timestep=1.0)
        self.assertEqual(len(result.steps), 61)  # 0 to 60s
        self.assertGreater(result.max_temperature, 25.0)

    # --------------------------------------------------------------------------
    # 7. Physical Relationships & Parameter Sensitivities
    # --------------------------------------------------------------------------

    def test_increasing_power_increases_steady_state(self) -> None:
        """Verify higher power produces higher steady state temperature."""
        p_low = self.model.steady_state_temperature(power=10.0)
        p_med = self.model.steady_state_temperature(power=25.0)
        p_high = self.model.steady_state_temperature(power=50.0)
        self.assertLess(p_low, p_med)
        self.assertLess(p_med, p_high)

    def test_increasing_r_th_increases_temperature_rise(self) -> None:
        """Verify larger thermal resistance increases steady state temperature rise."""
        model_low_r = ThermalModel(thermal_resistance=0.8, thermal_capacitance=20.0)
        model_high_r = ThermalModel(thermal_resistance=1.6, thermal_capacitance=20.0)

        rise_low = model_low_r.temperature_rise(power=20.0)
        rise_high = model_high_r.temperature_rise(power=20.0)
        self.assertLess(rise_low, rise_high)
        self.assertEqual(rise_high, 2.0 * rise_low)

    def test_increasing_c_th_slows_heating_rate(self) -> None:
        """Verify larger capacitance slows transient temperature rise without changing steady state."""
        model_fast = ThermalModel(thermal_resistance=1.0, thermal_capacitance=10.0)  # tau = 10s
        model_slow = ThermalModel(thermal_resistance=1.0, thermal_capacitance=50.0)  # tau = 50s

        # Steady-state is identical for both: 25 + 20*1 = 45 °C
        self.assertEqual(model_fast.steady_state_temperature(20.0), model_slow.steady_state_temperature(20.0))

        # At t = 10s, fast model will have heated much more than slow model
        res_fast = model_fast.simulate(power=20.0, duration=10.0, timestep=1.0)
        res_slow = model_slow.simulate(power=20.0, duration=10.0, timestep=1.0)

        self.assertGreater(res_fast.final_temperature, res_slow.final_temperature)

    # --------------------------------------------------------------------------
    # 8. Numerical Methods & Stability
    # --------------------------------------------------------------------------

    def test_integration_methods_consistency(self) -> None:
        """Test that EXACT, RK4, and EULER produce close results for fine timesteps."""
        dt = 0.05
        duration = 10.0
        power = 25.0

        res_exact = self.model.simulate(power=power, duration=duration, timestep=dt, method="EXACT")
        res_rk4 = self.model.simulate(power=power, duration=duration, timestep=dt, method="RK4")
        res_euler = self.model.simulate(power=power, duration=duration, timestep=dt, method="EULER")

        self.assertAlmostEqual(res_exact.final_temperature, res_rk4.final_temperature, places=3)
        self.assertAlmostEqual(res_exact.final_temperature, res_euler.final_temperature, places=1)

    def test_large_timestep_stability_with_exact_method(self) -> None:
        """Verify that default EXACT method remains unconditionally stable even for large timesteps."""
        # Large timestep (dt = 50s > tau = 24s)
        res_large_dt = self.model.simulate(power=30.0, duration=150.0, timestep=50.0, method="EXACT")
        self.assertFalse(math.isnan(res_large_dt.final_temperature))
        self.assertFalse(math.isinf(res_large_dt.final_temperature))
        self.assertLessEqual(res_large_dt.final_temperature, 61.0)

    # --------------------------------------------------------------------------
    # 9. Validation and Error Handling
    # --------------------------------------------------------------------------

    def test_negative_thermal_resistance_raises_error(self) -> None:
        """Test that negative R_th raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_resistance=-1.0)
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_resistance=0.0)

    def test_negative_thermal_capacitance_raises_error(self) -> None:
        """Test that non-positive C_th raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_capacitance=-5.0)
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_capacitance=0.0)

    def test_negative_timestep_raises_error(self) -> None:
        """Test that non-positive timestep raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=10.0, timestep=-1.0)
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=10.0, timestep=0.0)

    def test_negative_duration_raises_error(self) -> None:
        """Test that non-positive duration raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=-10.0, timestep=1.0)
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=0.0, timestep=1.0)

    def test_negative_power_raises_error(self) -> None:
        """Test that negative power raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=-5.0, duration=10.0, timestep=1.0)
        with self.assertRaises(InvalidThermalParameterError):
            self.model.steady_state_temperature(power=-1.0)

    def test_nan_and_inf_inputs_raise_error(self) -> None:
        """Test that NaN and Inf parameters are rejected."""
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(thermal_resistance=float("nan"))
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=float("nan"), duration=10.0, timestep=1.0)
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=float("inf"), timestep=1.0)

    def test_below_absolute_zero_temperature_raises_error(self) -> None:
        """Test that temperature below -273.15 °C raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            ThermalModel(ambient_temperature=-300.0)

    def test_unsupported_integration_method_raises_error(self) -> None:
        """Test that invalid integration method string raises InvalidThermalParameterError."""
        with self.assertRaises(InvalidThermalParameterError):
            self.model.simulate(power=10.0, duration=10.0, timestep=1.0, method="MAGIC_SOLVER")

    # --------------------------------------------------------------------------
    # 10. Structured Result Model Serialization & Indexing
    # --------------------------------------------------------------------------

    def test_result_structure_and_serialization(self) -> None:
        """Test indexing, length, and dictionary serialization of ThermalSimulationResult."""
        result = self.model.simulate(power=20.0, duration=10.0, timestep=1.0)
        self.assertEqual(len(result), 11)  # t = 0 to 10 inclusive

        step_0 = result[0]
        self.assertIsInstance(step_0, ThermalStepResult)
        self.assertEqual(step_0.time, 0.0)
        self.assertEqual(step_0.temperature, 25.0)

        # to_dict
        d = result.to_dict()
        self.assertIn("final_temperature", d)
        self.assertIn("steps", d)
        self.assertEqual(d["total_steps"], 11)
        self.assertEqual(d["ambient_temperature"], 25.0)


if __name__ == "__main__":
    unittest.main()
