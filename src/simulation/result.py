"""Structured simulation result representations and summary metrics."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterator, List, Optional

from src.power.config import PowerConfig
from src.power.power_result import PowerResult
from src.simulation.config import SimulationConfig
from src.simulation.exceptions import InvalidSimulationParameterError
from src.thermal.config import ThermalConfig
from src.thermal.thermal_result import ThermalStepResult


@dataclass(frozen=True)
class SimulationStepResult:
    """Represents the complete state of the pipeline at a discrete simulation timestep.

    Attributes:
        step_index: Discrete step index (0, 1, 2, ...).
        time: Simulation time in seconds.
        workload: Workload utilization ratio [0.0, 1.0].
        frequency: CPU operating clock frequency in Hertz (Hz).
        voltage: CPU operating supply voltage in Volts (V).
        dynamic_power: Dynamic power consumption in Watts (W).
        static_power: Static leakage power consumption in Watts (W).
        total_power: Total power consumption in Watts (W).
        temperature: Junction/core temperature at this step in °C.
        temperature_rise: Junction temperature elevation above ambient in °C.
        power_result: Underlying PowerResult instance from PowerEstimator.
        thermal_step: Underlying ThermalStepResult instance from ThermalModel.
    """

    step_index: int
    time: float
    workload: float
    frequency: float
    voltage: float
    dynamic_power: float
    static_power: float
    total_power: float
    temperature: float
    temperature_rise: float
    power_result: Optional[PowerResult] = None
    thermal_step: Optional[ThermalStepResult] = None

    def __post_init__(self) -> None:
        """Coerce numeric types."""
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "workload", float(self.workload))
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "voltage", float(self.voltage))
        object.__setattr__(self, "dynamic_power", float(self.dynamic_power))
        object.__setattr__(self, "static_power", float(self.static_power))
        object.__setattr__(self, "total_power", float(self.total_power))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "temperature_rise", float(self.temperature_rise))

    def to_dict(self) -> Dict[str, Any]:
        """Convert SimulationStepResult to dictionary representation."""
        return {
            "step_index": self.step_index,
            "time": round(self.time, 4),
            "workload": round(self.workload, 4),
            "voltage": round(self.voltage, 4),
            "frequency_hz": self.frequency,
            "frequency_ghz": round(self.frequency / 1.0e9, 4),
            "dynamic_power": round(self.dynamic_power, 6),
            "static_power": round(self.static_power, 6),
            "total_power": round(self.total_power, 6),
            "temperature": round(self.temperature, 4),
            "temperature_rise": round(self.temperature_rise, 4),
        }


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregated summary statistics derived from end-to-end simulation results.

    Attributes:
        peak_workload: Maximum workload utilization in simulation.
        peak_dynamic_power: Maximum dynamic power in Watts.
        peak_static_power: Maximum static power in Watts.
        peak_total_power: Maximum total power in Watts.
        peak_temperature: Maximum temperature reached in °C.
        final_temperature: Final recorded temperature at simulation end in °C.
        initial_temperature: Starting temperature at t=0 in °C.
        average_power: Time-average total power consumption in Watts.
        average_temperature: Time-average temperature in °C.
        time_at_peak_temperature: Timestamp in seconds when max temperature occurred.
        time_at_peak_power: Timestamp in seconds when max total power occurred.
        steady_state_final_estimate: Theoretical steady-state temperature for final power level in °C.
        duration: Total simulation duration in seconds.
        timestep: Simulation timestep delta_t in seconds.
        total_steps: Number of recorded simulation steps.
    """

    peak_workload: float
    peak_dynamic_power: float
    peak_static_power: float
    peak_total_power: float
    peak_temperature: float
    final_temperature: float
    initial_temperature: float
    average_power: float
    average_temperature: float
    time_at_peak_temperature: float
    time_at_peak_power: float
    steady_state_final_estimate: float
    duration: float
    timestep: float
    total_steps: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert SimulationSummary to dictionary format."""
        return {
            "peak_workload": round(self.peak_workload, 4),
            "peak_dynamic_power": round(self.peak_dynamic_power, 6),
            "peak_static_power": round(self.peak_static_power, 6),
            "peak_total_power": round(self.peak_total_power, 6),
            "peak_temperature": round(self.peak_temperature, 4),
            "final_temperature": round(self.final_temperature, 4),
            "initial_temperature": round(self.initial_temperature, 4),
            "average_power": round(self.average_power, 6),
            "average_temperature": round(self.average_temperature, 4),
            "time_at_peak_temperature": round(self.time_at_peak_temperature, 4),
            "time_at_peak_power": round(self.time_at_peak_power, 4),
            "steady_state_final_estimate": round(self.steady_state_final_estimate, 4),
            "duration": round(self.duration, 4),
            "timestep": round(self.timestep, 4),
            "total_steps": self.total_steps,
        }


@dataclass(frozen=True)
class ThermoShiftSimulationResult:
    """Comprehensive structured outcome of end-to-end Workload -> Power -> Temperature simulation.

    Attributes:
        times: List of simulation timestamp values [s].
        workloads: List of workload values [0.0 - 1.0].
        frequencies: List of CPU operating frequencies [Hz].
        voltages: List of operating supply voltages [V].
        dynamic_powers: List of dynamic power values [W].
        static_powers: List of static leakage power values [W].
        total_powers: List of total power values [W].
        temperatures: List of simulated temperatures [°C].
        steps: List of SimulationStepResult step objects.
        config: SimulationConfig instance.
        power_config: PowerConfig instance.
        thermal_config: ThermalConfig instance.
    """

    times: List[float]
    workloads: List[float]
    frequencies: List[float]
    voltages: List[float]
    dynamic_powers: List[float]
    static_powers: List[float]
    total_powers: List[float]
    temperatures: List[float]
    steps: List[SimulationStepResult]
    config: SimulationConfig
    power_config: PowerConfig
    thermal_config: ThermalConfig

    @property
    def final_temperature(self) -> float:
        """Final recorded temperature in °C."""
        return self.temperatures[-1] if self.temperatures else 25.0

    @property
    def max_temperature(self) -> float:
        """Peak temperature reached during simulation in °C."""
        return max(self.temperatures) if self.temperatures else 25.0

    @property
    def peak_temperature(self) -> float:
        """Alias for max_temperature in °C."""
        return self.max_temperature

    @property
    def min_temperature(self) -> float:
        """Lowest temperature recorded in °C."""
        return min(self.temperatures) if self.temperatures else 25.0

    @property
    def final_power(self) -> float:
        """Final total power consumption in Watts."""
        return self.total_powers[-1] if self.total_powers else 0.0

    @property
    def max_power(self) -> float:
        """Peak total power reached during simulation in Watts."""
        return max(self.total_powers) if self.total_powers else 0.0

    @property
    def peak_total_power(self) -> float:
        """Alias for max_power in Watts."""
        return self.max_power

    @property
    def average_power(self) -> float:
        """Time-average total power consumption in Watts."""
        return sum(self.total_powers) / len(self.total_powers) if self.total_powers else 0.0

    @property
    def average_temperature(self) -> float:
        """Time-average junction temperature in °C."""
        return sum(self.temperatures) / len(self.temperatures) if self.temperatures else 25.0

    def summary(self) -> SimulationSummary:
        """Calculate and return comprehensive summary statistics for the simulation."""
        if not self.steps:
            raise InvalidSimulationParameterError("Cannot calculate summary for empty simulation result")

        max_temp = max(self.temperatures)
        peak_temp_idx = self.temperatures.index(max_temp)
        time_peak_temp = self.times[peak_temp_idx]

        max_p_tot = max(self.total_powers)
        peak_power_idx = self.total_powers.index(max_p_tot)
        time_peak_power = self.times[peak_power_idx]

        avg_power = sum(self.total_powers) / len(self.total_powers)
        avg_temp = sum(self.temperatures) / len(self.temperatures)

        final_p_tot = self.total_powers[-1]
        steady_state_final = self.thermal_config.ambient_temperature + (
            final_p_tot * self.thermal_config.thermal_resistance
        )

        return SimulationSummary(
            peak_workload=max(self.workloads),
            peak_dynamic_power=max(self.dynamic_powers),
            peak_static_power=max(self.static_powers),
            peak_total_power=max_p_tot,
            peak_temperature=max_temp,
            final_temperature=self.temperatures[-1],
            initial_temperature=self.temperatures[0],
            average_power=avg_power,
            average_temperature=avg_temp,
            time_at_peak_temperature=time_peak_temp,
            time_at_peak_power=time_peak_power,
            steady_state_final_estimate=steady_state_final,
            duration=self.config.duration,
            timestep=self.config.timestep,
            total_steps=len(self.steps),
        )

    def analyze_safety(self, monitor: Optional[Any] = None) -> Any:
        """Run thermal safety analysis on this simulation result using ThermalSafetyMonitor.

        Args:
            monitor: Optional ThermalSafetyMonitor instance. If None, default ThermalSafetyMonitor is used.

        Returns:
            ThermalSafetyAnalysis instance containing aggregated safety metrics and step classifications.
        """
        from src.safety.monitor import ThermalSafetyMonitor
        safety_mon = monitor if monitor is not None else ThermalSafetyMonitor()
        return safety_mon.analyze(self)

    def analyze(self, analyzer: Optional[Any] = None, safety_monitor: Optional[Any] = None) -> Any:
        """Run comprehensive statistical and safety analysis on this simulation result.

        Args:
            analyzer: Optional SimulationAnalyzer instance.
            safety_monitor: Optional ThermalSafetyMonitor instance.

        Returns:
            SimulationAnalysisSummary object containing comprehensive statistics.
        """
        from src.analysis.analyzer import SimulationAnalyzer
        anz = analyzer if analyzer is not None else SimulationAnalyzer()
        return anz.analyze(self, safety_monitor=safety_monitor)

    def plot_workload(self, save_path: Optional[str] = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot CPU workload utilization time series."""
        from src.visualization.plotter import plot_workload
        return plot_workload(self, save_path=save_path, show=show, **kwargs)

    def plot_power(self, save_path: Optional[str] = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot dynamic, static, and total power time series."""
        from src.visualization.plotter import plot_power
        return plot_power(self, save_path=save_path, show=show, **kwargs)

    def plot_temperature(self, save_path: Optional[str] = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot temperature evolution with thermal limit threshold overlays."""
        from src.visualization.plotter import plot_temperature
        return plot_temperature(self, save_path=save_path, show=show, **kwargs)

    def plot_overview(self, save_path: Optional[str] = None, show: bool = False, **kwargs: Any) -> Any:
        """Plot combined overview dashboard (Workload, Power, Temperature)."""
        from src.visualization.plotter import plot_overview
        return plot_overview(self, save_path=save_path, show=show, **kwargs)

    def get_variable(self, name: str) -> List[float]:
        """Access a specific output variable series by name.

        Supported names: 'time', 'workload', 'frequency', 'voltage', 'dynamic_power', 'static_power', 'total_power', 'temperature'.
        """
        key = name.lower().strip()
        mapping = {
            "time": self.times,
            "times": self.times,
            "workload": self.workloads,
            "workloads": self.workloads,
            "frequency": self.frequencies,
            "frequencies": self.frequencies,
            "voltage": self.voltages,
            "voltages": self.voltages,
            "dynamic_power": self.dynamic_powers,
            "static_power": self.static_powers,
            "total_power": self.total_powers,
            "power": self.total_powers,
            "temperature": self.temperatures,
            "temperatures": self.temperatures,
        }
        if key not in mapping:
            raise KeyError(
                f"Unknown variable name '{name}'. Valid names: {list(mapping.keys())}"
            )
        return list(mapping[key])

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire ThermoShiftSimulationResult to serializable dictionary."""
        return {
            "summary": self.summary().to_dict(),
            "config": {
                "duration": self.config.duration,
                "timestep": self.config.timestep,
                "integration_method": self.config.integration_method,
            },
            "steps": [s.to_dict() for s in self.steps],
        }

    def plot(self, save_path: Optional[str] = None, show: bool = True) -> Any:
        """Plot Workload, Power, and Temperature vs Time curves if matplotlib is available.

        Args:
            save_path: Optional file path to save plot figure image.
            show: Whether to display figure with plt.show().

        Returns:
            Matplotlib figure object if available, else None.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Plotting skipped.")
            return None

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # 1. Workload
        ax1.plot(self.times, [w * 100 for w in self.workloads], color="tab:blue", linewidth=2, label="Workload (%)")
        ax1.set_ylabel("Workload (%)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_title("ThermoShift End-to-End Simulation Results")
        ax1.legend(loc="upper left")

        # 2. Power
        ax2.plot(self.times, self.total_powers, color="tab:red", linewidth=2, label="Total Power (W)")
        ax2.plot(self.times, self.dynamic_powers, color="tab:orange", linestyle="--", label="Dynamic Power (W)")
        ax2.plot(self.times, self.static_powers, color="tab:purple", linestyle=":", label="Static Power (W)")
        ax2.set_ylabel("Power (W)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

        # 3. Temperature
        ax3.plot(self.times, self.temperatures, color="tab:red", linewidth=2, label="Temperature (°C)")
        ax3.axhline(self.thermal_config.ambient_temperature, color="tab:gray", linestyle=":", label="Ambient Temp")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Temperature (°C)")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        if show:
            plt.show()

        return fig

    def __len__(self) -> int:
        """Return total number of recorded steps."""
        return len(self.steps)

    def __iter__(self) -> Iterator[SimulationStepResult]:
        """Iterate over individual simulation step results."""
        return iter(self.steps)

    def __getitem__(self, index: int) -> SimulationStepResult:
        """Index access to simulation step results."""
        return self.steps[index]
