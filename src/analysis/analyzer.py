"""ThermoShift Simulation Analyzer Subsystem."""

import math
from typing import Any, Dict, List, Optional

from src.analysis.exceptions import InvalidAnalysisError
from src.analysis.summary import SimulationAnalysisSummary
from src.safety.monitor import ThermalSafetyMonitor
from src.simulation.result import ThermoShiftSimulationResult


class SimulationAnalyzer:
    """Statistical and safety analysis engine for ThermoShift simulation results.

    Consumes structured ThermoShiftSimulationResult output to calculate statistical metrics,
    thermal safety violation statistics, and state durations without recalculating physical equations.
    """

    def analyze(
        self,
        results: ThermoShiftSimulationResult,
        safety_monitor: Optional[ThermalSafetyMonitor] = None,
    ) -> SimulationAnalysisSummary:
        """Perform comprehensive statistical and safety analysis on simulation results.

        Args:
            results: ThermoShiftSimulationResult object from end-to-end simulation.
            safety_monitor: Optional ThermalSafetyMonitor instance. If omitted, default monitor is used.

        Returns:
            SimulationAnalysisSummary dataclass containing aggregated insights.

        Raises:
            InvalidAnalysisError: If results input is missing, empty, or contains invalid data.
        """
        self.validate_results(results)

        times = results.times
        workloads = results.workloads
        dyn_powers = results.dynamic_powers
        stat_powers = results.static_powers
        tot_powers = results.total_powers
        temperatures = results.temperatures
        timestep = results.config.timestep
        duration = results.config.duration

        # 1. Workload Statistics
        peak_w = float(max(workloads))
        avg_w = float(sum(workloads) / len(workloads))
        min_w = float(min(workloads))

        # 2. Power Statistics
        peak_dyn_p = float(max(dyn_powers))
        peak_stat_p = float(max(stat_powers))
        peak_tot_p = float(max(tot_powers))
        avg_dyn_p = float(sum(dyn_powers) / len(dyn_powers))
        avg_stat_p = float(sum(stat_powers) / len(stat_powers))
        avg_tot_p = float(sum(tot_powers) / len(tot_powers))
        min_tot_p = float(min(tot_powers))

        # 3. Temperature Statistics
        peak_temp = float(max(temperatures))
        peak_idx = temperatures.index(peak_temp)
        time_peak_temp = float(times[peak_idx])
        avg_temp = float(sum(temperatures) / len(temperatures))
        min_temp = float(min(temperatures))
        final_temp = float(temperatures[-1])

        # 4. Thermal Safety Analysis Integration
        monitor = safety_monitor if safety_monitor is not None else ThermalSafetyMonitor()
        safety_analysis = monitor.analyze(results)

        warning_occurred = (
            safety_analysis.warning_timesteps > 0
            or safety_analysis.critical_timesteps > 0
            or safety_analysis.overheating_timesteps > 0
        )
        critical_occurred = (
            safety_analysis.critical_timesteps > 0
            or safety_analysis.overheating_timesteps > 0
        )
        overheating_occurred = safety_analysis.has_overheating

        time_above_warning = (
            safety_analysis.warning_duration
            + safety_analysis.critical_duration
            + safety_analysis.overheating_duration
        )
        time_above_critical = (
            safety_analysis.critical_duration
            + safety_analysis.overheating_duration
        )
        time_above_overheating = safety_analysis.overheating_duration

        return SimulationAnalysisSummary(
            peak_workload=peak_w,
            average_workload=avg_w,
            min_workload=min_w,
            peak_dynamic_power=peak_dyn_p,
            peak_static_power=peak_stat_p,
            peak_total_power=peak_tot_p,
            average_dynamic_power=avg_dyn_p,
            average_static_power=avg_stat_p,
            average_total_power=avg_tot_p,
            min_total_power=min_tot_p,
            peak_temperature=peak_temp,
            time_of_peak_temperature=time_peak_temp,
            average_temperature=avg_temp,
            min_temperature=min_temp,
            final_temperature=final_temp,
            warning_occurred=warning_occurred,
            critical_occurred=critical_occurred,
            overheating_occurred=overheating_occurred,
            first_warning_time=safety_analysis.first_warning_time,
            first_critical_time=safety_analysis.first_critical_time,
            first_overheating_time=safety_analysis.first_overheating_time,
            normal_timesteps=safety_analysis.normal_timesteps,
            warning_timesteps=safety_analysis.warning_timesteps,
            critical_timesteps=safety_analysis.critical_timesteps,
            overheating_timesteps=safety_analysis.overheating_timesteps,
            normal_duration=safety_analysis.normal_duration,
            warning_duration=safety_analysis.warning_duration,
            critical_duration=safety_analysis.critical_duration,
            overheating_duration=safety_analysis.overheating_duration,
            total_time_above_warning=time_above_warning,
            total_time_above_critical=time_above_critical,
            total_time_above_overheating=time_above_overheating,
            duration=duration,
            timestep=timestep,
            total_steps=len(results.steps),
        )

    def validate_results(self, results: ThermoShiftSimulationResult) -> None:
        """Validate result object structure, field consistency, and numerical validity.

        Raises:
            InvalidAnalysisError: If validation fails.
        """
        if not isinstance(results, ThermoShiftSimulationResult):
            raise InvalidAnalysisError(
                f"Expected ThermoShiftSimulationResult object, got {type(results).__name__}"
            )

        if not results.steps:
            raise InvalidAnalysisError("Simulation result contains no steps")

        n_steps = len(results.steps)

        for name, series in [
            ("times", results.times),
            ("workloads", results.workloads),
            ("frequencies", results.frequencies),
            ("voltages", results.voltages),
            ("dynamic_powers", results.dynamic_powers),
            ("static_powers", results.static_powers),
            ("total_powers", results.total_powers),
            ("temperatures", results.temperatures),
        ]:
            if not series:
                raise InvalidAnalysisError(f"Simulation result field '{name}' is empty")
            if len(series) != len(results.times):
                raise InvalidAnalysisError(
                    f"Array length mismatch: '{name}' has length {len(series)}, expected {len(results.times)}"
                )

        # Check numeric validity (finite float)
        for i in range(len(results.times)):
            t = results.times[i]
            w = results.workloads[i]
            p = results.total_powers[i]
            temp = results.temperatures[i]

            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                raise InvalidAnalysisError(f"Invalid timestamp at step {i}: {t}")

            if i > 0 and t < results.times[i - 1]:
                raise InvalidAnalysisError(
                    f"Non-monotonic timestamp sequence: t[{i}]={t} < t[{i-1}]={results.times[i-1]}"
                )

            if not isinstance(w, (int, float)) or not math.isfinite(w):
                raise InvalidAnalysisError(f"Invalid workload value at step {i}: {w}")

            if not isinstance(p, (int, float)) or not math.isfinite(p):
                raise InvalidAnalysisError(f"Invalid total power value at step {i}: {p}")

            if not isinstance(temp, (int, float)) or not math.isfinite(temp):
                raise InvalidAnalysisError(f"Invalid temperature value at step {i}: {temp}")
