"""ThermoShift Visualization Subsystem — Workload, Power, Temperature, and Dashboard Plotting."""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.analysis.analyzer import SimulationAnalyzer
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.safety.result import ThermalSafetyAnalysis
from src.simulation.result import ThermoShiftSimulationResult
from src.visualization.exceptions import InvalidVisualizationError, VisualizationError


def _get_plt() -> Any:
    """Helper to safely import matplotlib.pyplot."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise VisualizationError(
            "matplotlib is required for ThermoShift visualization functions. Please install matplotlib."
        ) from exc


def plot_workload(
    results: ThermoShiftSimulationResult,
    title: str = "Workload vs Time",
    figsize: Tuple[float, float] = (10, 4),
    as_percentage: bool = True,
    color: str = "tab:blue",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """Plot CPU Workload utilization over time.

    Args:
        results: ThermoShiftSimulationResult instance.
        title: Plot title.
        figsize: Figure width and height tuple in inches.
        as_percentage: If True, Y-axis displayed as percentage [0-100%], else ratio [0.0-1.0].
        color: Plot line color.
        save_path: Optional file path to export image (e.g., PNG, SVG).
        show: If True, call plt.show().

    Returns:
        Tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    analyzer = SimulationAnalyzer()
    try:
        analyzer.validate_results(results)
    except Exception as exc:
        raise InvalidVisualizationError(f"Cannot plot workload from invalid result: {exc}") from exc

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    times = results.times
    workloads = [w * 100.0 if as_percentage else w for w in results.workloads]
    unit_str = "%" if as_percentage else "Ratio"

    ax.plot(times, workloads, color=color, linewidth=2, label=f"Workload ({unit_str})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Workload ({unit_str})")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    if as_percentage:
        ax.set_ylim(-2, 105)
    else:
        ax.set_ylim(-0.02, 1.05)

    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax


def plot_power(
    results: ThermoShiftSimulationResult,
    title: str = "Power Dissipation vs Time",
    figsize: Tuple[float, float] = (10, 4),
    show_components: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """Plot CPU Power consumption (Dynamic, Static, Total) over time.

    Args:
        results: ThermoShiftSimulationResult instance.
        title: Plot title.
        figsize: Figure width and height tuple.
        show_components: If True, plot dynamic and static power components alongside total power.
        save_path: Optional file path to export image.
        show: If True, call plt.show().

    Returns:
        Tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    analyzer = SimulationAnalyzer()
    try:
        analyzer.validate_results(results)
    except Exception as exc:
        raise InvalidVisualizationError(f"Cannot plot power from invalid result: {exc}") from exc

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    times = results.times
    ax.plot(times, results.total_powers, color="tab:red", linewidth=2.5, label="Total Power (W)")

    if show_components:
        ax.plot(
            times,
            results.dynamic_powers,
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="Dynamic Power (W)",
        )
        ax.plot(
            times,
            results.static_powers,
            color="tab:purple",
            linestyle=":",
            linewidth=1.8,
            label="Static Power (W)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax


def plot_temperature(
    results: ThermoShiftSimulationResult,
    safety_config: Optional[ThermalSafetyConfig] = None,
    safety_analysis: Optional[ThermalSafetyAnalysis] = None,
    title: str = "Temperature Evolution & Safety Limits vs Time",
    figsize: Tuple[float, float] = (10, 5),
    show_thresholds: bool = True,
    show_violations: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """Plot CPU Core Temperature over time with optional thermal limit overlays and violation markers.

    Args:
        results: ThermoShiftSimulationResult instance.
        safety_config: Optional ThermalSafetyConfig. If omitted, default ThermalSafetyConfig is used.
        safety_analysis: Optional ThermalSafetyAnalysis. If omitted, generated automatically.
        title: Plot title.
        figsize: Figure size tuple.
        show_thresholds: If True, overlay horizontal warning, critical, and maximum/overheating lines.
        show_violations: If True, mark threshold crossing events directly on temperature line.
        save_path: Optional export file path.
        show: If True, call plt.show().

    Returns:
        Tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    analyzer = SimulationAnalyzer()
    try:
        analyzer.validate_results(results)
    except Exception as exc:
        raise InvalidVisualizationError(f"Cannot plot temperature from invalid result: {exc}") from exc

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    times = results.times
    temps = results.temperatures

    # Main temperature curve
    ax.plot(times, temps, color="tab:red", linewidth=2.2, label="Junction Temp (°C)")

    # Ambient baseline
    ax.axhline(
        results.thermal_config.ambient_temperature,
        color="tab:gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Ambient ({results.thermal_config.ambient_temperature:.1f} °C)",
    )

    s_cfg = safety_config if safety_config is not None else ThermalSafetyConfig()

    if show_thresholds:
        ax.axhline(
            s_cfg.warning_temperature,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Warning ({s_cfg.warning_temperature:.1f} °C)",
        )
        ax.axhline(
            s_cfg.critical_temperature,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=f"Critical ({s_cfg.critical_temperature:.1f} °C)",
        )
        ax.axhline(
            s_cfg.maximum_temperature,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Overheating ({s_cfg.maximum_temperature:.1f} °C)",
        )

    if show_violations:
        monitor = ThermalSafetyMonitor(config=s_cfg)
        analysis = (
            safety_analysis
            if safety_analysis is not None
            else monitor.analyze(results)
        )

        for event in analysis.transitions:
            if "ENTERED" in event.event_type.value:
                ax.plot(
                    event.time,
                    event.temperature,
                    marker="o",
                    markersize=8,
                    color="red" if "OVERHEATING" in event.event_type.value else "darkorange",
                    label=f"Event: {event.event_type.value}",
                )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.85)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax


def plot_overview(
    results: ThermoShiftSimulationResult,
    safety_config: Optional[ThermalSafetyConfig] = None,
    safety_analysis: Optional[ThermalSafetyAnalysis] = None,
    title: str = "ThermoShift End-to-End Simulation Dashboard",
    figsize: Tuple[float, float] = (10, 9),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Plot combined 3-panel dashboard overview: Workload, Power, and Temperature vs Time.

    Args:
        results: ThermoShiftSimulationResult instance.
        safety_config: Optional ThermalSafetyConfig.
        safety_analysis: Optional ThermalSafetyAnalysis.
        title: Overall dashboard title.
        figsize: Dashboard figure size.
        save_path: Optional export file path.
        show: If True, call plt.show().

    Returns:
        Tuple of (matplotlib.figure.Figure, (ax_workload, ax_power, ax_temperature)).
    """
    analyzer = SimulationAnalyzer()
    try:
        analyzer.validate_results(results)
    except Exception as exc:
        raise InvalidVisualizationError(f"Cannot plot overview from invalid result: {exc}") from exc

    plt = _get_plt()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    times = results.times
    workloads = [w * 100.0 for w in results.workloads]

    # Panel 1: Workload
    ax1.plot(times, workloads, color="tab:blue", linewidth=2, label="Workload (%)")
    ax1.set_ylabel("Workload (%)")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_ylim(-2, 105)

    # Panel 2: Power
    ax2.plot(times, results.total_powers, color="tab:red", linewidth=2.2, label="Total Power (W)")
    ax2.plot(times, results.dynamic_powers, color="tab:orange", linestyle="--", label="Dynamic Power (W)")
    ax2.plot(times, results.static_powers, color="tab:purple", linestyle=":", label="Static Power (W)")
    ax2.set_ylabel("Power (W)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # Panel 3: Temperature & Safety Thresholds
    ax3.plot(times, results.temperatures, color="tab:red", linewidth=2.2, label="Junction Temp (°C)")
    ax3.axhline(
        results.thermal_config.ambient_temperature,
        color="tab:gray",
        linestyle=":",
        label=f"Ambient ({results.thermal_config.ambient_temperature:.1f} °C)",
    )

    s_cfg = safety_config if safety_config is not None else ThermalSafetyConfig()
    ax3.axhline(s_cfg.warning_temperature, color="orange", linestyle="--", label=f"Warning ({s_cfg.warning_temperature:.1f} °C)")
    ax3.axhline(s_cfg.critical_temperature, color="darkorange", linestyle="--", label=f"Critical ({s_cfg.critical_temperature:.1f} °C)")
    ax3.axhline(s_cfg.maximum_temperature, color="red", linestyle="--", label=f"Overheating ({s_cfg.maximum_temperature:.1f} °C)")

    # Violation Markers
    monitor = ThermalSafetyMonitor(config=s_cfg)
    analysis = (
        safety_analysis
        if safety_analysis is not None
        else monitor.analyze(results)
    )
    for event in analysis.transitions:
        if "ENTERED" in event.event_type.value:
            ax3.plot(
                event.time,
                event.temperature,
                marker="o",
                markersize=7,
                color="red" if "OVERHEATING" in event.event_type.value else "darkorange",
            )

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", framealpha=0.85)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, (ax1, ax2, ax3)


class SimulationPlotter:
    """Class wrapper providing convenient object-oriented plotting methods."""

    def __init__(self, safety_config: Optional[ThermalSafetyConfig] = None) -> None:
        """Initialize SimulationPlotter with optional safety configuration."""
        self.safety_config = safety_config

    def plot_workload(
        self,
        results: ThermoShiftSimulationResult,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[Any, Any]:
        """Plot workload time series."""
        return plot_workload(results, save_path=save_path, show=show)

    def plot_power(
        self,
        results: ThermoShiftSimulationResult,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[Any, Any]:
        """Plot power time series."""
        return plot_power(results, save_path=save_path, show=show)

    def plot_temperature(
        self,
        results: ThermoShiftSimulationResult,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[Any, Any]:
        """Plot temperature time series with thermal limit overlays."""
        return plot_temperature(results, safety_config=self.safety_config, save_path=save_path, show=show)

    def plot_overview(
        self,
        results: ThermoShiftSimulationResult,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[Any, Tuple[Any, Any, Any]]:
        """Plot combined overview dashboard."""
        return plot_overview(results, safety_config=self.safety_config, save_path=save_path, show=show)
