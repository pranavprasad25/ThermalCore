"""Structured analysis summary dataclass representing statistical simulation insights."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SimulationAnalysisSummary:
    """Comprehensive statistical and safety summary of a ThermoShift simulation run.

    Workload Metrics:
        peak_workload: Maximum workload utilization ratio [0.0, 1.0].
        average_workload: Average workload utilization ratio [0.0, 1.0].
        min_workload: Minimum workload utilization ratio [0.0, 1.0].

    Power Metrics:
        peak_dynamic_power: Maximum dynamic power in Watts.
        peak_static_power: Maximum static leakage power in Watts.
        peak_total_power: Maximum total power in Watts.
        average_dynamic_power: Average dynamic power in Watts.
        average_static_power: Average static leakage power in Watts.
        average_total_power: Average total power in Watts.
        min_total_power: Minimum total power in Watts.

    Temperature Metrics:
        peak_temperature: Maximum core temperature in °C.
        time_of_peak_temperature: Timestamp of peak core temperature in seconds.
        average_temperature: Average core temperature in °C.
        min_temperature: Minimum core temperature in °C.
        final_temperature: Final core temperature at simulation end in °C.

    Thermal Safety & Violation Metrics:
        warning_occurred: True if temperature crossed warning threshold.
        critical_occurred: True if temperature crossed critical threshold.
        overheating_occurred: True if temperature reached or exceeded maximum/overheating threshold.
        first_warning_time: Timestamp of first warning event in seconds (if any).
        first_critical_time: Timestamp of first critical event in seconds (if any).
        first_overheating_time: Timestamp of first overheating event in seconds (if any).
        normal_timesteps: Count of timesteps in NORMAL state.
        warning_timesteps: Count of timesteps in WARNING state.
        critical_timesteps: Count of timesteps in CRITICAL state.
        overheating_timesteps: Count of timesteps in OVERHEATING state.
        normal_duration: Total duration in NORMAL state in seconds.
        warning_duration: Total duration in WARNING state in seconds.
        critical_duration: Total duration in CRITICAL state in seconds.
        overheating_duration: Total duration in OVERHEATING state in seconds.
        total_time_above_warning: Total time spent at or above warning threshold in seconds.
        total_time_above_critical: Total time spent at or above critical threshold in seconds.
        total_time_above_overheating: Total time spent at or above maximum/overheating threshold in seconds.

    Simulation Details:
        duration: Total simulation duration in seconds.
        timestep: Simulation timestep delta_t in seconds.
        total_steps: Number of simulation steps.
    """

    # Workload
    peak_workload: float
    average_workload: float
    min_workload: float

    # Power
    peak_dynamic_power: float
    peak_static_power: float
    peak_total_power: float
    average_dynamic_power: float
    average_static_power: float
    average_total_power: float
    min_total_power: float

    # Temperature
    peak_temperature: float
    time_of_peak_temperature: float
    average_temperature: float
    min_temperature: float
    final_temperature: float

    # Thermal Safety
    warning_occurred: bool
    critical_occurred: bool
    overheating_occurred: bool
    first_warning_time: Optional[float]
    first_critical_time: Optional[float]
    first_overheating_time: Optional[float]
    normal_timesteps: int
    warning_timesteps: int
    critical_timesteps: int
    overheating_timesteps: int
    normal_duration: float
    warning_duration: float
    critical_duration: float
    overheating_duration: float
    total_time_above_warning: float
    total_time_above_critical: float
    total_time_above_overheating: float

    # Simulation Details
    duration: float
    timestep: float
    total_steps: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis summary into a serializable dictionary."""
        return {
            "workload": {
                "peak_workload": round(self.peak_workload, 4),
                "peak_workload_pct": round(self.peak_workload * 100, 2),
                "average_workload": round(self.average_workload, 4),
                "average_workload_pct": round(self.average_workload * 100, 2),
                "min_workload": round(self.min_workload, 4),
                "min_workload_pct": round(self.min_workload * 100, 2),
            },
            "power": {
                "peak_dynamic_power": round(self.peak_dynamic_power, 4),
                "peak_static_power": round(self.peak_static_power, 4),
                "peak_total_power": round(self.peak_total_power, 4),
                "average_dynamic_power": round(self.average_dynamic_power, 4),
                "average_static_power": round(self.average_static_power, 4),
                "average_total_power": round(self.average_total_power, 4),
                "min_total_power": round(self.min_total_power, 4),
            },
            "temperature": {
                "peak_temperature": round(self.peak_temperature, 4),
                "time_of_peak_temperature": round(self.time_of_peak_temperature, 4),
                "average_temperature": round(self.average_temperature, 4),
                "min_temperature": round(self.min_temperature, 4),
                "final_temperature": round(self.final_temperature, 4),
            },
            "thermal_safety": {
                "warning_occurred": self.warning_occurred,
                "critical_occurred": self.critical_occurred,
                "overheating_occurred": self.overheating_occurred,
                "first_warning_time": round(self.first_warning_time, 4) if self.first_warning_time is not None else None,
                "first_critical_time": round(self.first_critical_time, 4) if self.first_critical_time is not None else None,
                "first_overheating_time": round(self.first_overheating_time, 4) if self.first_overheating_time is not None else None,
                "normal_timesteps": self.normal_timesteps,
                "warning_timesteps": self.warning_timesteps,
                "critical_timesteps": self.critical_timesteps,
                "overheating_timesteps": self.overheating_timesteps,
                "normal_duration": round(self.normal_duration, 4),
                "warning_duration": round(self.warning_duration, 4),
                "critical_duration": round(self.critical_duration, 4),
                "overheating_duration": round(self.overheating_duration, 4),
                "total_time_above_warning": round(self.total_time_above_warning, 4),
                "total_time_above_critical": round(self.total_time_above_critical, 4),
                "total_time_above_overheating": round(self.total_time_above_overheating, 4),
            },
            "simulation": {
                "duration": round(self.duration, 4),
                "timestep": round(self.timestep, 4),
                "total_steps": self.total_steps,
            },
        }

    def to_string(self) -> str:
        """Format analysis summary into a human-readable text report."""
        lines = [
            "=" * 60,
            " ThermoShift Simulation Analysis Summary",
            "=" * 60,
            f"Simulation Duration: {self.duration:.2f} s | Timestep: {self.timestep:.2f} s | Total Steps: {self.total_steps}",
            "-" * 60,
            "WORKLOAD:",
            f"  - Peak Workload:      {self.peak_workload * 100:.1f} %",
            f"  - Average Workload:   {self.average_workload * 100:.1f} %",
            f"  - Minimum Workload:   {self.min_workload * 100:.1f} %",
            "-" * 60,
            "POWER:",
            f"  - Peak Total Power:   {self.peak_total_power:.4f} W",
            f"  - Peak Dynamic Power: {self.peak_dynamic_power:.4f} W",
            f"  - Peak Static Power:  {self.peak_static_power:.4f} W",
            f"  - Average Power:      {self.average_total_power:.4f} W",
            f"  - Minimum Power:      {self.min_total_power:.4f} W",
            "-" * 60,
            "TEMPERATURE:",
            f"  - Peak Temperature:   {self.peak_temperature:.2f} °C (at t = {self.time_of_peak_temperature:.1f} s)",
            f"  - Average Temp:       {self.average_temperature:.2f} °C",
            f"  - Minimum Temp:       {self.min_temperature:.2f} °C",
            f"  - Final Temp:         {self.final_temperature:.2f} °C",
            "-" * 60,
            "THERMAL SAFETY STATUS:",
            f"  - Warning Occurred:     {self.warning_occurred} (First: {f'{self.first_warning_time:.1f}s' if self.first_warning_time is not None else 'N/A'})",
            f"  - Critical Occurred:    {self.critical_occurred} (First: {f'{self.first_critical_time:.1f}s' if self.first_critical_time is not None else 'N/A'})",
            f"  - Overheating Occurred: {self.overheating_occurred} (First: {f'{self.first_overheating_time:.1f}s' if self.first_overheating_time is not None else 'N/A'})",
            f"  - Duration > Warning:   {self.total_time_above_warning:.1f} s",
            f"  - Duration > Critical:  {self.total_time_above_critical:.1f} s",
            f"  - Duration > Maximum:   {self.total_time_above_overheating:.1f} s",
            "=" * 60,
        ]
        return "\n".join(lines)
