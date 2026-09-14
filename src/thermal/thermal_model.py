"""First-order Thermal RC Model for steady-state and transient temperature simulation."""

from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.thermal.config import ThermalConfig
from src.thermal.exceptions import (
    InvalidThermalParameterError,
    ThermalModelError,
    ThermalSimulationError,
)
from src.thermal.thermal_result import ThermalSimulationResult, ThermalStepResult


class ThermalModel:
    """First-order Thermal RC equivalent model of a CPU core or processor package.

    Governing Differential Equation:
        C_th * dT/dt = P(t) - (T(t) - T_ambient) / R_th

    Equivalently:
        dT/dt = (P(t) * R_th - (T(t) - T_ambient)) / tau
        where tau = R_th * C_th (thermal time constant in seconds).

    Steady-State Equation:
        T_steady = T_ambient + (P * R_th)
    """

    def __init__(
        self,
        thermal_resistance: Optional[float] = None,
        thermal_capacitance: Optional[float] = None,
        ambient_temperature: Optional[float] = None,
        initial_temperature: Optional[float] = None,
        config: Optional[ThermalConfig] = None,
    ) -> None:
        """Initialize the Thermal Model.

        Args:
            thermal_resistance: Thermal resistance R_th in °C/W.
            thermal_capacitance: Thermal capacitance C_th in J/°C.
            ambient_temperature: Ambient environment temperature in °C.
            initial_temperature: Starting junction temperature in °C.
            config: Optional ThermalConfig instance to draw defaults from.
        """
        cfg = config if config is not None else ThermalConfig()

        self._r_th = float(thermal_resistance if thermal_resistance is not None else cfg.thermal_resistance)
        self._c_th = float(thermal_capacitance if thermal_capacitance is not None else cfg.thermal_capacitance)
        self._t_ambient = float(ambient_temperature if ambient_temperature is not None else cfg.ambient_temperature)
        self._t_initial = float(initial_temperature if initial_temperature is not None else (
            cfg.initial_temperature if initial_temperature is None and thermal_resistance is None else self._t_ambient
        ))
        if initial_temperature is not None:
            self._t_initial = float(initial_temperature)

        self._validate_physical_params(self._r_th, self._c_th, self._t_ambient, self._t_initial)
        self._current_temperature: float = self._t_initial

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def thermal_resistance(self) -> float:
        """Thermal resistance R_th in °C/W."""
        return self._r_th

    @property
    def thermal_capacitance(self) -> float:
        """Thermal capacitance C_th in J/°C."""
        return self._c_th

    @property
    def ambient_temperature(self) -> float:
        """Ambient surrounding temperature in °C."""
        return self._t_ambient

    @property
    def initial_temperature(self) -> float:
        """Initial baseline temperature in °C."""
        return self._t_initial

    @property
    def time_constant(self) -> float:
        """Thermal time constant tau = R_th * C_th in seconds."""
        return self._r_th * self._c_th

    @property
    def tau(self) -> float:
        """Alias for thermal time constant tau in seconds."""
        return self.time_constant

    @property
    def current_temperature(self) -> float:
        """Current tracked junction temperature in °C."""
        return self._current_temperature

    # --------------------------------------------------------------------------
    # Steady-State Calculations
    # --------------------------------------------------------------------------

    def temperature_rise(
        self,
        power: Union[float, int, Any],
        thermal_resistance: Optional[float] = None,
    ) -> float:
        """Calculate steady-state temperature rise: Delta_T = P * R_th.

        Args:
            power: Dissipated power in Watts (W) or PowerResult object.
            thermal_resistance: Optional override for thermal resistance R_th in °C/W.

        Returns:
            Temperature elevation above ambient in °C.
        """
        p_val = self._extract_power_value(power)
        r_th = thermal_resistance if thermal_resistance is not None else self._r_th
        self._validate_r_th(r_th)
        return float(p_val * r_th)

    def steady_state_temperature(
        self,
        power: Union[float, int, Any],
        ambient_temperature: Optional[float] = None,
        thermal_resistance: Optional[float] = None,
    ) -> float:
        """Calculate steady-state temperature: T_steady = T_ambient + (P * R_th).

        Args:
            power: Dissipated power in Watts (W) or PowerResult object.
            ambient_temperature: Optional override for ambient temperature in °C.
            thermal_resistance: Optional override for thermal resistance R_th in °C/W.

        Returns:
            Equilibrium steady-state temperature in °C.
        """
        p_val = self._extract_power_value(power)
        t_amb = ambient_temperature if ambient_temperature is not None else self._t_ambient
        r_th = thermal_resistance if thermal_resistance is not None else self._r_th
        self._validate_temperature(t_amb, "ambient_temperature")
        self._validate_r_th(r_th)
        return float(t_amb + (p_val * r_th))

    # --------------------------------------------------------------------------
    # Discrete Step Integration
    # --------------------------------------------------------------------------

    def step(
        self,
        power: Union[float, int, Any],
        dt: float,
        method: str = "EXACT",
    ) -> float:
        """Advance internal temperature by timestep dt under applied power.

        Args:
            power: Power in Watts or PowerResult object.
            dt: Timestep duration in seconds (must be > 0).
            method: Integration method ('EXACT', 'EULER', or 'RK4').

        Returns:
            New updated temperature in °C.
        """
        p_val = self._extract_power_value(power)
        self._validate_dt(dt)

        self._current_temperature = self._integrate_step(
            t_curr=self._current_temperature,
            power=p_val,
            dt=dt,
            method=method,
        )
        return self._current_temperature

    def reset(self, initial_temperature: Optional[float] = None) -> None:
        """Reset internal temperature state.

        Args:
            initial_temperature: Starting temperature in °C (defaults to model initial_temperature).
        """
        if initial_temperature is not None:
            self._validate_temperature(initial_temperature, "initial_temperature")
            self._current_temperature = float(initial_temperature)
        else:
            self._current_temperature = self._t_initial

    # --------------------------------------------------------------------------
    # Transient Simulation
    # --------------------------------------------------------------------------

    def simulate(
        self,
        power: Union[float, int, Any, Callable[[float], Union[float, Any]], Sequence[Union[float, Any]]],
        duration: float,
        timestep: float,
        initial_temperature: Optional[float] = None,
        method: str = "EXACT",
    ) -> ThermalSimulationResult:
        """Run a transient thermal simulation over specified duration and timestep.

        Args:
            power: Power input. Can be:
                   - A constant float/int/PowerResult
                   - A callable f(t) -> power
                   - A sequence of power values
            duration: Total duration to simulate in seconds (must be > 0).
            timestep: Simulation time step in seconds (must be > 0).
            initial_temperature: Starting temperature override in °C.
            method: Integration method ('EXACT', 'EULER', or 'RK4').

        Returns:
            ThermalSimulationResult containing complete trajectory data and summaries.
        """
        self._validate_duration(duration)
        self._validate_dt(timestep)

        t_start = float(initial_temperature if initial_temperature is not None else self._t_initial)
        self._validate_temperature(t_start, "initial_temperature")

        # Number of steps
        n_steps = int(math.ceil(duration / timestep))
        if n_steps <= 0:
            raise InvalidThermalParameterError(f"Calculated number of steps ({n_steps}) must be positive.")

        times: List[float] = [0.0]
        temperatures: List[float] = [t_start]
        powers: List[float] = []
        steps: List[ThermalStepResult] = []

        curr_t = t_start
        curr_time = 0.0

        for i in range(n_steps):
            # Determine power for this step
            p_val = self._resolve_power_at_time(power, curr_time, i, n_steps)
            if i == 0:
                powers.append(p_val)
                steps.append(
                    ThermalStepResult(
                        time=0.0,
                        temperature=t_start,
                        power=p_val,
                        temperature_rise=t_start - self._t_ambient,
                    )
                )

            # Integrate forward by timestep
            curr_t = self._integrate_step(curr_t, p_val, timestep, method)
            curr_time = (i + 1) * timestep

            times.append(curr_time)
            temperatures.append(curr_t)
            next_p = self._resolve_power_at_time(power, curr_time, i + 1, n_steps)
            powers.append(next_p)
            steps.append(
                ThermalStepResult(
                    time=curr_time,
                    temperature=curr_t,
                    power=next_p,
                    temperature_rise=curr_t - self._t_ambient,
                )
            )

        return ThermalSimulationResult(
            times=times,
            temperatures=temperatures,
            powers=powers,
            steps=steps,
            ambient_temperature=self._t_ambient,
            initial_temperature=t_start,
            thermal_resistance=self._r_th,
            thermal_capacitance=self._c_th,
            time_constant=self.time_constant,
            duration=duration,
            timestep=timestep,
        )

    def simulate_profile(
        self,
        power_profile: Sequence[Union[float, Any]],
        timestep: float,
        initial_temperature: Optional[float] = None,
        method: str = "EXACT",
    ) -> ThermalSimulationResult:
        """Simulate thermal response across a sequence of discrete power profile samples.

        Args:
            power_profile: Sequence of power samples applied at each timestep interval.
            timestep: Duration of each power sample in seconds.
            initial_temperature: Starting temperature override in °C.
            method: Integration method ('EXACT', 'EULER', 'RK4').

        Returns:
            ThermalSimulationResult instance.
        """
        if not power_profile:
            raise InvalidThermalParameterError("power_profile sequence cannot be empty.")
        duration = len(power_profile) * timestep
        return self.simulate(
            power=power_profile,
            duration=duration,
            timestep=timestep,
            initial_temperature=initial_temperature,
            method=method,
        )

    # --------------------------------------------------------------------------
    # Numerical Integration Helpers
    # --------------------------------------------------------------------------

    def _integrate_step(self, t_curr: float, power: float, dt: float, method: str = "EXACT") -> float:
        """Advance temperature by dt using specified integration method."""
        method_upper = method.upper()

        if method_upper in ("EXACT", "ANALYTICAL"):
            # Exact analytical exponential solution for 1st-order linear ODE with constant power over dt:
            # T(t + dt) = T_steady + (T(t) - T_steady) * exp(-dt / tau)
            t_steady = self._t_ambient + (power * self._r_th)
            tau = self.time_constant
            decay = math.exp(-dt / tau)
            t_next = t_steady + ((t_curr - t_steady) * decay)
            return float(t_next)

        elif method_upper == "EULER":
            # Forward Euler: dT/dt = (P - (T - Tamb)/Rth) / Cth
            # T_next = T + (dT/dt) * dt
            tau = self.time_constant
            dt_dt = (power * self._r_th - (t_curr - self._t_ambient)) / tau
            t_next = t_curr + (dt_dt * dt)
            if not math.isfinite(t_next):
                raise ThermalSimulationError(f"Euler integration diverged with dt={dt}, tau={tau}")
            return float(t_next)

        elif method_upper == "RK4":
            # 4th-order Runge-Kutta
            tau = self.time_constant
            r_th = self._r_th
            t_amb = self._t_ambient

            def f(temp: float) -> float:
                return (power * r_th - (temp - t_amb)) / tau

            k1 = f(t_curr)
            k2 = f(t_curr + 0.5 * dt * k1)
            k3 = f(t_curr + 0.5 * dt * k2)
            k4 = f(t_curr + dt * k3)

            t_next = t_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return float(t_next)

        else:
            raise InvalidThermalParameterError(
                f"Unsupported integration method '{method}'. Supported methods: ('EXACT', 'EULER', 'RK4')"
            )

    def _resolve_power_at_time(
        self,
        power_spec: Union[float, int, Any, Callable[[float], Union[float, Any]], Sequence[Union[float, Any]]],
        time: float,
        step_index: int,
        total_steps: int,
    ) -> float:
        """Resolve numeric power value at given time or index."""
        if callable(power_spec):
            try:
                val = power_spec(time)
                return self._extract_power_value(val)
            except Exception as exc:
                raise InvalidThermalParameterError(f"Error evaluating power function at time={time}: {exc}") from exc
        elif isinstance(power_spec, (list, tuple)):
            if len(power_spec) == 0:
                raise InvalidThermalParameterError("Power sequence cannot be empty.")
            idx = min(step_index, len(power_spec) - 1)
            return self._extract_power_value(power_spec[idx])
        else:
            return self._extract_power_value(power_spec)

    def _extract_power_value(self, power: Any) -> float:
        """Extract and validate numeric power in Watts."""
        if hasattr(power, "total_power"):
            val = power.total_power
        else:
            val = power

        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            raise InvalidThermalParameterError(f"Power must be a finite number in Watts, got {power}")
        if val < 0.0:
            raise InvalidThermalParameterError(f"Power cannot be negative, got {val} W")
        return float(val)

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------

    def _validate_physical_params(self, r_th: float, c_th: float, t_amb: float, t_init: float) -> None:
        self._validate_r_th(r_th)
        self._validate_c_th(c_th)
        self._validate_temperature(t_amb, "ambient_temperature")
        self._validate_temperature(t_init, "initial_temperature")

    def _validate_r_th(self, r_th: Any) -> None:
        if not isinstance(r_th, (int, float)) or isinstance(r_th, bool) or not math.isfinite(r_th) or r_th <= 0.0:
            raise InvalidThermalParameterError(f"Thermal resistance R_th must be a positive finite float, got {r_th}")

    def _validate_c_th(self, c_th: Any) -> None:
        if not isinstance(c_th, (int, float)) or isinstance(c_th, bool) or not math.isfinite(c_th) or c_th <= 0.0:
            raise InvalidThermalParameterError(f"Thermal capacitance C_th must be a positive finite float, got {c_th}")

    def _validate_temperature(self, temp: Any, param_name: str = "temperature") -> None:
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or not math.isfinite(temp):
            raise InvalidThermalParameterError(f"{param_name} must be a finite number in °C, got {temp}")
        if temp < -273.15:
            raise InvalidThermalParameterError(f"{param_name} cannot be below absolute zero (-273.15 °C), got {temp}")

    def _validate_dt(self, dt: Any) -> None:
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0.0:
            raise InvalidThermalParameterError(f"Timestep must be a positive finite float in seconds, got {dt}")

    def _validate_duration(self, duration: Any) -> None:
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0.0:
            raise InvalidThermalParameterError(f"Duration must be a positive finite float in seconds, got {duration}")
