"""ThermoShift End-to-End Simulation Engine and Orchestrator."""

from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from src.cpu.dvfs import LinearDVFSMapper, OperatingConditionMapper
from src.cpu.operating_condition import OperatingCondition
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.power.power_result import PowerResult
from src.simulation.config import SimulationConfig
from src.simulation.exceptions import InvalidSimulationParameterError, SimulationError
from src.simulation.result import (
    SimulationStepResult,
    SimulationSummary,
    ThermoShiftSimulationResult,
)
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.thermal.thermal_result import ThermalStepResult
from src.workload.profile import WorkloadPoint, WorkloadProfile


class ThermoShiftSimulation:
    """Orchestrates end-to-end Workload -> Operating Conditions -> Power -> Temperature simulation.

    Pipeline execution sequence at each simulation timestep t_k:
        1. Read workload: w(t_k) from WorkloadProfile
        2. Map operating conditions: (V, f) = DVFSMapper.map_workload(w(t_k))
        3. Estimate power: PowerResult = PowerEstimator.estimate(w(t_k), V, f, T(t_k))
        4. Advance thermal model: T(t_k + dt) = ThermalModel.step(PowerResult, dt)
        5. Record structured step result and iterate until duration complete.
    """

    def __init__(
        self,
        power_estimator: Optional[PowerEstimator] = None,
        thermal_model: Optional[ThermalModel] = None,
        dvfs_mapper: Optional[OperatingConditionMapper] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        """Initialize ThermoShift end-to-end simulation engine.

        Args:
            power_estimator: PowerEstimator instance (default created if None).
            thermal_model: ThermalModel instance (default created if None).
            dvfs_mapper: OperatingConditionMapper instance (default LinearDVFSMapper if None).
            config: SimulationConfig instance (default created if None).
        """
        self.power_estimator: PowerEstimator = power_estimator if power_estimator is not None else PowerEstimator()
        self.thermal_model: ThermalModel = thermal_model if thermal_model is not None else ThermalModel()
        self.dvfs_mapper: OperatingConditionMapper = (
            dvfs_mapper if dvfs_mapper is not None else LinearDVFSMapper(power_config=self.power_estimator.config)
        )
        self.config: SimulationConfig = config if config is not None else SimulationConfig()

    def run(
        self,
        workload_profile: Union[WorkloadProfile, float, int, Sequence[Union[Tuple[float, float], WorkloadPoint]], Callable[[float], float]],
        duration: Optional[float] = None,
        timestep: Optional[float] = None,
        initial_temperature: Optional[float] = None,
        ambient_temperature: Optional[float] = None,
    ) -> ThermoShiftSimulationResult:
        """Run complete end-to-end simulation over specified duration and timestep.

        Args:
            workload_profile: Workload profile representation. Can be:
                - WorkloadProfile instance
                - Constant float/int workload value [0.0, 1.0]
                - Sequence of (time, workload) or (duration, workload) tuples
                - Callable function f(t) -> workload
            duration: Override for simulation duration in seconds.
            timestep: Override for simulation timestep delta_t in seconds.
            initial_temperature: Starting temperature override in °C.
            ambient_temperature: Ambient temperature override in °C.

        Returns:
            ThermoShiftSimulationResult with complete time series data and summary metrics.

        Raises:
            InvalidSimulationParameterError: If any parameter or workload is invalid.
        """
        if duration is not None:
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0.0:
                raise InvalidSimulationParameterError(f"Simulation duration must be > 0, got {duration}")

        if timestep is not None:
            if not isinstance(timestep, (int, float)) or not math.isfinite(timestep) or timestep <= 0.0:
                raise InvalidSimulationParameterError(f"Simulation timestep must be > 0, got {timestep}")

        # Resolve WorkloadProfile instance
        profile = self._resolve_workload_profile(workload_profile, duration)

        # Resolve duration and timestep
        sim_duration = float(duration if duration is not None else (
            profile.duration if profile.duration > 0 else self.config.duration
        ))
        sim_timestep = float(timestep if timestep is not None else self.config.timestep)

        # Validate duration & timestep
        if not math.isfinite(sim_duration) or sim_duration <= 0.0:
            raise InvalidSimulationParameterError(f"Simulation duration must be > 0, got {sim_duration}")

        if not math.isfinite(sim_timestep) or sim_timestep <= 0.0:
            raise InvalidSimulationParameterError(f"Simulation timestep must be > 0, got {sim_timestep}")

        if sim_timestep > sim_duration:
            raise InvalidSimulationParameterError(
                f"Timestep ({sim_timestep}s) cannot be greater than duration ({sim_duration}s)"
            )

        # Resolve temperature parameters
        amb_temp = float(
            ambient_temperature
            if ambient_temperature is not None
            else (
                self.config.ambient_temperature
                if self.config.ambient_temperature is not None
                else self.thermal_model.ambient_temperature
            )
        )
        init_temp = float(
            initial_temperature
            if initial_temperature is not None
            else (
                self.config.initial_temperature
                if self.config.initial_temperature is not None
                else self.thermal_model.initial_temperature
            )
        )

        if not math.isfinite(amb_temp):
            raise InvalidSimulationParameterError(f"Ambient temperature must be finite, got {amb_temp}")
        if not math.isfinite(init_temp):
            raise InvalidSimulationParameterError(f"Initial temperature must be finite, got {init_temp}")

        # Update / verify thermal model configuration for this run
        self.thermal_model.reset(initial_temperature=init_temp)

        # Calculate discrete simulation time steps
        n_steps = int(round(sim_duration / sim_timestep))
        if n_steps <= 0:
            raise InvalidSimulationParameterError(f"Calculated simulation steps ({n_steps}) must be positive")

        times: List[float] = []
        workloads: List[float] = []
        frequencies: List[float] = []
        voltages: List[float] = []
        dynamic_powers: List[float] = []
        static_powers: List[float] = []
        total_powers: List[float] = []
        temperatures: List[float] = []
        steps: List[SimulationStepResult] = []

        curr_temp = init_temp

        for k in range(n_steps + 1):
            t_k = round(k * sim_timestep, 10)
            if t_k > sim_duration:
                t_k = sim_duration

            # 1. Read workload at time t_k
            w_k = profile.get_workload(t_k)
            if not (0.0 <= w_k <= 1.0):
                raise InvalidSimulationParameterError(
                    f"Sampled workload at t={t_k}s is out of bounds [0.0, 1.0]: {w_k}"
                )

            # 2. Determine operating conditions from workload
            op_cond = self.dvfs_mapper.map_workload(w_k)

            # 3. Pass operating conditions to PowerEstimator
            power_res = self.power_estimator.estimate(
                workload=w_k,
                voltage=op_cond.voltage,
                frequency=op_cond.frequency,
                temperature=curr_temp,
                activity_factor=op_cond.activity_factor,
            )

            # Record current state at t_k
            times.append(t_k)
            workloads.append(w_k)
            frequencies.append(op_cond.frequency)
            voltages.append(op_cond.voltage)
            dynamic_powers.append(power_res.dynamic_power)
            static_powers.append(power_res.static_power)
            total_powers.append(power_res.total_power)
            temperatures.append(curr_temp)

            step_res = SimulationStepResult(
                step_index=k,
                time=t_k,
                workload=w_k,
                frequency=op_cond.frequency,
                voltage=op_cond.voltage,
                dynamic_power=power_res.dynamic_power,
                static_power=power_res.static_power,
                total_power=power_res.total_power,
                temperature=curr_temp,
                temperature_rise=curr_temp - amb_temp,
                power_result=power_res,
            )
            steps.append(step_res)

            # 4. Feed total power to ThermalModel to advance temperature to t_{k+1}
            if k < n_steps:
                curr_temp = self.thermal_model.step(
                    power=power_res,
                    dt=sim_timestep,
                    method=self.config.integration_method,
                )

        run_config = SimulationConfig(
            duration=sim_duration,
            timestep=sim_timestep,
            ambient_temperature=amb_temp,
            initial_temperature=init_temp,
            integration_method=self.config.integration_method,
        )

        thermal_cfg = ThermalConfig(
            ambient_temperature=amb_temp,
            initial_temperature=init_temp,
            thermal_resistance=self.thermal_model.thermal_resistance,
            thermal_capacitance=self.thermal_model.thermal_capacitance,
        )

        return ThermoShiftSimulationResult(
            times=times,
            workloads=workloads,
            frequencies=frequencies,
            voltages=voltages,
            dynamic_powers=dynamic_powers,
            static_powers=static_powers,
            total_powers=total_powers,
            temperatures=temperatures,
            steps=steps,
            config=run_config,
            power_config=self.power_estimator.config,
            thermal_config=thermal_cfg,
        )

    def _resolve_workload_profile(
        self,
        workload_input: Any,
        duration: Optional[float] = None,
    ) -> WorkloadProfile:
        """Helper to convert loose workload input types into a validated WorkloadProfile instance."""
        if isinstance(workload_input, WorkloadProfile):
            if workload_input.is_empty():
                raise InvalidSimulationParameterError("Provided WorkloadProfile is empty")
            return workload_input

        if isinstance(workload_input, (int, float)):
            w_val = float(workload_input)
            if not (0.0 <= w_val <= 1.0):
                raise InvalidSimulationParameterError(f"Constant workload must be in [0.0, 1.0], got {w_val}")
            dur = float(duration if duration is not None else self.config.duration)
            return WorkloadProfile.from_constant(workload=w_val, duration=dur)

        if callable(workload_input):
            dur = float(duration if duration is not None else self.config.duration)
            return WorkloadProfile.from_function(func=workload_input, duration=dur, dt=self.config.timestep)

        if isinstance(workload_input, (list, tuple)):
            if not workload_input:
                raise InvalidSimulationParameterError("Workload sequence input cannot be empty")
            return WorkloadProfile.from_points(points=workload_input)

        raise InvalidSimulationParameterError(
            f"Unsupported workload_profile input type: {type(workload_input).__name__}"
        )


# Alias for backward compatibility or alternate naming
SimulationEngine = ThermoShiftSimulation
