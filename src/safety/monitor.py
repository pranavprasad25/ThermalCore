"""Thermal Safety Monitor and Time-Series Analyzer."""

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from src.safety.config import ThermalSafetyConfig
from src.safety.exceptions import InvalidThermalSafetyParameterError, ThermalSafetyError
from src.safety.result import (
    ThermalSafetyAnalysis,
    ThermalSafetyResult,
    ThermalTransitionEvent,
)
from src.safety.status import ThermalStatus, ThermalTransitionEventType
from src.simulation.result import ThermoShiftSimulationResult


class ThermalSafetyMonitor:
    """Thermal Safety Subsystem Monitor.

    Monitors temperatures against configured thermal limits, classifies safety states,
    detects threshold crossing events, and analyzes simulation time-series results.

    Flow:
        Temperature(t)
              ↓
        Thermal Safety Monitor
              ↓
        Compare Against Limits
              ↓
        Thermal Status
              ↓
        NORMAL / WARNING / CRITICAL / OVERHEATING
    """

    def __init__(
        self,
        config: Optional[ThermalSafetyConfig] = None,
        warning_temperature: Optional[float] = None,
        critical_temperature: Optional[float] = None,
        maximum_temperature: Optional[float] = None,
        hysteresis: float = 0.0,
    ) -> None:
        """Initialize ThermalSafetyMonitor with optional configuration or direct limit overrides.

        Args:
            config: Optional ThermalSafetyConfig instance.
            warning_temperature: Override for warning threshold in °C.
            critical_temperature: Override for critical threshold in °C.
            maximum_temperature: Override for maximum/overheating threshold in °C.
            hysteresis: Override for de-escalation hysteresis offset in °C.
        """
        if config is not None:
            w_temp = warning_temperature if warning_temperature is not None else config.warning_temperature
            c_temp = critical_temperature if critical_temperature is not None else config.critical_temperature
            m_temp = maximum_temperature if maximum_temperature is not None else config.maximum_temperature
            h_val = hysteresis if hysteresis != 0.0 else config.hysteresis
            self.config = ThermalSafetyConfig(
                warning_temperature=w_temp,
                critical_temperature=c_temp,
                maximum_temperature=m_temp,
                hysteresis=h_val,
            )
        else:
            w_temp = warning_temperature if warning_temperature is not None else 70.0
            c_temp = critical_temperature if critical_temperature is not None else 85.0
            m_temp = maximum_temperature if maximum_temperature is not None else 95.0
            self.config = ThermalSafetyConfig(
                warning_temperature=w_temp,
                critical_temperature=c_temp,
                maximum_temperature=m_temp,
                hysteresis=hysteresis,
            )

    def _validate_temperature_input(self, temperature: float) -> float:
        """Validate input temperature type, finiteness, and physical bounds."""
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            raise InvalidThermalSafetyParameterError(
                f"Temperature input must be a finite float, got {temperature}"
            )
        temp = float(temperature)
        if temp < -100.0 or temp > 300.0:
            raise InvalidThermalSafetyParameterError(
                f"Temperature value out of physically valid bounds [-100.0, 300.0] °C: {temp}"
            )
        return temp

    def classify_status(
        self,
        temperature: float,
        previous_status: Optional[ThermalStatus] = None,
    ) -> ThermalStatus:
        """Classify a temperature value into ThermalStatus (NORMAL, WARNING, CRITICAL, OVERHEATING).

        Applies hysteresis when de-escalating state severity if hysteresis > 0.

        Args:
            temperature: Current core temperature in °C.
            previous_status: Previous ThermalStatus for hysteresis evaluation.

        Returns:
            ThermalStatus enum value.
        """
        t = self._validate_temperature_input(temperature)
        h = self.config.hysteresis

        # 1. Evaluate OVERHEATING boundary
        if t >= self.config.maximum_temperature:
            return ThermalStatus.OVERHEATING

        # Hysteresis check for de-escalation from OVERHEATING
        if previous_status == ThermalStatus.OVERHEATING and h > 0.0:
            if t > (self.config.maximum_temperature - h):
                return ThermalStatus.OVERHEATING

        # 2. Evaluate CRITICAL boundary
        if t >= self.config.critical_temperature:
            return ThermalStatus.CRITICAL

        # Hysteresis check for de-escalation from CRITICAL
        if previous_status == ThermalStatus.CRITICAL and h > 0.0:
            if t > (self.config.critical_temperature - h):
                return ThermalStatus.CRITICAL

        # 3. Evaluate WARNING boundary
        if t >= self.config.warning_temperature:
            return ThermalStatus.WARNING

        # Hysteresis check for de-escalation from WARNING
        if previous_status == ThermalStatus.WARNING and h > 0.0:
            if t > (self.config.warning_temperature - h):
                return ThermalStatus.WARNING

        # 4. Default safe state
        return ThermalStatus.NORMAL

    def check_temperature(
        self,
        temperature: float,
        timestamp: Optional[float] = None,
        step_index: Optional[int] = None,
        previous_status: Optional[ThermalStatus] = None,
    ) -> ThermalSafetyResult:
        """Evaluate a single temperature value against safety limits.

        Args:
            temperature: Core junction temperature in °C.
            timestamp: Optional evaluation timestamp in seconds.
            step_index: Optional step index.
            previous_status: Optional previous ThermalStatus for hysteresis.

        Returns:
            ThermalSafetyResult object encapsulating temperature, status, and boolean flags.
        """
        status = self.classify_status(temperature, previous_status=previous_status)
        return ThermalSafetyResult(
            temperature=float(temperature),
            status=status,
            is_safe=(status == ThermalStatus.NORMAL),
            is_warning=(status == ThermalStatus.WARNING),
            is_critical=(status == ThermalStatus.CRITICAL),
            is_overheating=(status == ThermalStatus.OVERHEATING),
            timestamp=float(timestamp) if timestamp is not None else None,
            step_index=int(step_index) if step_index is not None else None,
        )

    def is_overheating(self, temperature: float) -> bool:
        """Explicit check if temperature meets or exceeds maximum/overheating threshold."""
        return self.check_temperature(temperature).is_overheating

    def is_safe(self, temperature: float) -> bool:
        """Explicit check if temperature is safe (status == NORMAL)."""
        return self.check_temperature(temperature).is_safe

    def is_warning(self, temperature: float) -> bool:
        """Explicit check if temperature is in WARNING state."""
        return self.check_temperature(temperature).is_warning

    def is_critical(self, temperature: float) -> bool:
        """Explicit check if temperature is in CRITICAL state."""
        return self.check_temperature(temperature).is_critical

    def _determine_transition_event(
        self,
        step_index: int,
        time: float,
        temperature: float,
        previous_status: ThermalStatus,
        new_status: ThermalStatus,
    ) -> Optional[ThermalTransitionEvent]:
        """Determine transition event type if a status boundary was crossed."""
        if previous_status == new_status:
            return None

        event_type = ThermalTransitionEventType.NO_CHANGE

        if new_status == ThermalStatus.WARNING and previous_status == ThermalStatus.NORMAL:
            event_type = ThermalTransitionEventType.WARNING_ENTERED
        elif new_status == ThermalStatus.CRITICAL and previous_status in (ThermalStatus.NORMAL, ThermalStatus.WARNING):
            event_type = ThermalTransitionEventType.CRITICAL_ENTERED
        elif new_status == ThermalStatus.OVERHEATING:
            event_type = ThermalTransitionEventType.OVERHEATING_ENTERED
        elif new_status == ThermalStatus.CRITICAL and previous_status == ThermalStatus.OVERHEATING:
            event_type = ThermalTransitionEventType.OVERHEATING_CLEARED
        elif new_status == ThermalStatus.WARNING and previous_status in (ThermalStatus.CRITICAL, ThermalStatus.OVERHEATING):
            event_type = ThermalTransitionEventType.CRITICAL_CLEARED
        elif new_status == ThermalStatus.NORMAL and previous_status in (ThermalStatus.WARNING, ThermalStatus.CRITICAL, ThermalStatus.OVERHEATING):
            event_type = ThermalTransitionEventType.WARNING_CLEARED

        return ThermalTransitionEvent(
            step_index=step_index,
            time=time,
            temperature=temperature,
            previous_status=previous_status,
            new_status=new_status,
            event_type=event_type,
        )

    def analyze(self, simulation_results: ThermoShiftSimulationResult) -> ThermalSafetyAnalysis:
        """Analyze a complete simulation result time-series for thermal safety violations and transitions.

        Args:
            simulation_results: ThermoShiftSimulationResult instance from Day 3 Task 3 simulation.

        Returns:
            ThermalSafetyAnalysis object containing aggregated metrics, transition events, and step flags.
        """
        if not isinstance(simulation_results, ThermoShiftSimulationResult):
            raise InvalidThermalSafetyParameterError(
                f"Expected ThermoShiftSimulationResult input, got {type(simulation_results).__name__}"
            )

        if not simulation_results.steps:
            raise InvalidThermalSafetyParameterError("Cannot analyze empty simulation result")

        times = simulation_results.times
        temperatures = simulation_results.temperatures
        timestep = simulation_results.config.timestep

        max_temp = max(temperatures)
        peak_idx = temperatures.index(max_temp)
        time_peak = times[peak_idx]

        step_results: List[ThermalSafetyResult] = []
        transitions: List[ThermalTransitionEvent] = []

        normal_count = 0
        warning_count = 0
        critical_count = 0
        overheating_count = 0

        first_warning_time: Optional[float] = None
        first_critical_time: Optional[float] = None
        first_overheating_time: Optional[float] = None

        prev_status: Optional[ThermalStatus] = None

        for k in range(len(times)):
            t_k = times[k]
            temp_k = temperatures[k]

            res = self.check_temperature(
                temperature=temp_k,
                timestamp=t_k,
                step_index=k,
                previous_status=prev_status,
            )
            step_results.append(res)

            # State counts
            if res.status == ThermalStatus.NORMAL:
                normal_count += 1
            elif res.status == ThermalStatus.WARNING:
                warning_count += 1
                if first_warning_time is None:
                    first_warning_time = t_k
            elif res.status == ThermalStatus.CRITICAL:
                critical_count += 1
                if first_critical_time is None:
                    first_critical_time = t_k
            elif res.status == ThermalStatus.OVERHEATING:
                overheating_count += 1
                if first_overheating_time is None:
                    first_overheating_time = t_k

            # Transition detection
            if prev_status is not None and prev_status != res.status:
                event = self._determine_transition_event(
                    step_index=k,
                    time=t_k,
                    temperature=temp_k,
                    previous_status=prev_status,
                    new_status=res.status,
                )
                if event:
                    transitions.append(event)

            prev_status = res.status

        has_violations = (warning_count + critical_count + overheating_count) > 0
        has_overheating = overheating_count > 0

        return ThermalSafetyAnalysis(
            maximum_temperature=max_temp,
            time_of_maximum_temperature=time_peak,
            final_temperature=temperatures[-1],
            total_timesteps=len(step_results),
            duration=simulation_results.config.duration,
            normal_timesteps=normal_count,
            warning_timesteps=warning_count,
            critical_timesteps=critical_count,
            overheating_timesteps=overheating_count,
            normal_duration=normal_count * timestep,
            warning_duration=warning_count * timestep,
            critical_duration=critical_count * timestep,
            overheating_duration=overheating_count * timestep,
            has_violations=has_violations,
            has_overheating=has_overheating,
            first_warning_time=first_warning_time,
            first_critical_time=first_critical_time,
            first_overheating_time=first_overheating_time,
            transitions=transitions,
            step_results=step_results,
            config=self.config,
        )
