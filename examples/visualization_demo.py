"""ThermoShift Day 3 Task 5 — Visualization & Analysis Subsystem Demo."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.analyzer import SimulationAnalyzer
from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.simulation.config import SimulationConfig
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.visualization.plotter import (
    plot_overview,
    plot_power,
    plot_temperature,
    plot_workload,
)
from src.workload.profile import WorkloadProfile


def main() -> None:
    """Run runnable Visualization & Analysis subsystem demonstration."""
    print("=" * 85)
    print(" ThermoShift — Visualization & Statistical Analysis Subsystem Demo")
    print("=" * 85)

    # 1. Define Realistic Workload Profile
    timeline = [
        (0.0, 0.20),   # 0s  - 20%  (Idle)
        (4.0, 0.70),   # 4s  - 70%  (Startup Load)
        (10.0, 1.00),  # 10s - 100% (Turbo Compute Burst)
        (22.0, 0.40),  # 22s - 40%  (Throttled Load)
        (35.0, 0.10),  # 35s - 10%  (Recovery / Cooldown)
    ]
    workload_profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

    # 2. Configure Power & Thermal Subsystems
    power_config = PowerConfig(
        capacitance=2.8e-9,           # 2.8 nF
        base_static_power=10.0,       # 10.0 W base leakage
        min_voltage=0.8,
        max_voltage=1.40,             # 1.40 V max Turbo voltage
        min_frequency=1.0e9,
        max_frequency=3.8e9,          # 3.8 GHz max clock
        reference_temperature=25.0,
        temperature_coefficient=0.018,
        static_power_model="LINEAR",
    )
    power_estimator = PowerEstimator(config=power_config)

    thermal_config = ThermalConfig(
        ambient_temperature=25.0,
        initial_temperature=25.0,
        thermal_resistance=1.8,       # 1.8 °C/W
        thermal_capacitance=12.0,     # tau = 21.6s
    )
    thermal_model = ThermalModel(config=thermal_config)
    dvfs_mapper = LinearDVFSMapper(power_config=power_config)

    simulation = ThermoShiftSimulation(
        power_estimator=power_estimator,
        thermal_model=thermal_model,
        dvfs_mapper=dvfs_mapper,
        config=SimulationConfig(duration=50.0, timestep=1.0),
    )

    # 3. Execute End-to-End Simulation
    print("\n[1] Running Simulation Pipeline (0.0s -> 50.0s, dt=1.0s)...")
    results = simulation.run(workload_profile=workload_profile, duration=50.0)

    # 4. Configure Safety Limits & Perform Statistical Analysis
    safety_config = ThermalSafetyConfig(
        warning_temperature=45.0,
        critical_temperature=52.0,
        maximum_temperature=57.0,
    )
    safety_monitor = ThermalSafetyMonitor(config=safety_config)

    analyzer = SimulationAnalyzer()
    analysis_summary = analyzer.analyze(results, safety_monitor=safety_monitor)

    print("\n[2] Statistical & Safety Analysis Results:")
    print(analysis_summary.to_string())

    # 5. Generate and Export Visualizations
    out_dir = os.path.dirname(__file__)
    file_workload = os.path.join(out_dir, "plot_workload.png")
    file_power = os.path.join(out_dir, "plot_power.png")
    file_temperature = os.path.join(out_dir, "plot_temperature.png")
    file_overview = os.path.join(out_dir, "plot_overview.png")

    print("\n[3] Generating and Exporting Plots...")
    fig_w, _ = plot_workload(results, save_path=file_workload, show=False)
    print(f"    - Workload Plot exported to:    {file_workload}")

    fig_p, _ = plot_power(results, save_path=file_power, show=False)
    print(f"    - Power Plot exported to:       {file_power}")

    fig_t, _ = plot_temperature(
        results,
        safety_config=safety_config,
        save_path=file_temperature,
        show=False,
    )
    print(f"    - Temperature Plot exported to: {file_temperature}")

    fig_o, _ = plot_overview(
        results,
        safety_config=safety_config,
        save_path=file_overview,
        show=False,
    )
    print(f"    - Dashboard Plot exported to:   {file_overview}")

    print("\n" + "=" * 85)
    print(" Visualization and Analysis demonstration completed successfully!")
    print("=" * 85)


if __name__ == "__main__":
    main()
