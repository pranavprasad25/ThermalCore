"""ThermoShift Day 3 — Master End-to-End Pipeline Demonstration Script.

Demonstrates complete Workload -> DVFS -> Power -> Thermal -> Safety -> Analysis -> Visualization pipeline.
"""

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
from src.visualization.plotter import SimulationPlotter
from src.workload.profile import WorkloadProfile


def main() -> None:
    """Execute complete end-to-end ThermoShift simulation pipeline demonstration."""
    print("=" * 95)
    print(" ThermoShift — Day 3 End-to-End Simulation, Thermal Safety & Visualization Master Demo")
    print("=" * 95)

    # 1. Define Workload Profile
    print("\n[Step 1] Defining Time-Varying CPU Workload Profile...")
    timeline = [
        (0.0, 0.20),   # 0-2s: Low idle load (20%)
        (2.0, 0.60),   # 2-5s: Medium startup load (60%)
        (5.0, 1.00),   # 5-12s: Heavy compute burst (100%)
        (12.0, 0.30),  # 12-18s: Cooldown load (30%)
        (18.0, 0.80),  # 18-25s: Steady compute (80%)
    ]
    workload_profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")
    print(f"  - Duration       : {workload_profile.duration:.1f} s")
    print(f"  - Peak Workload  : {workload_profile.max_workload * 100:.1f} %")
    print(f"  - Timeline Steps : {timeline}")

    # 2. Configure Power Model
    print("\n[Step 2] Configuring CPU Power Estimation Subsystem...")
    power_config = PowerConfig(
        capacitance=1.5e-9,           # 1.5 nF effective switching capacitance
        base_static_power=5.0,        # 5.0 W baseline leakage power
        min_voltage=0.8,              # 0.8 V
        max_voltage=1.25,             # 1.25 V
        min_frequency=1.0e9,          # 1.0 GHz
        max_frequency=3.5e9,          # 3.5 GHz
        reference_temperature=25.0,   # 25.0 °C
        temperature_coefficient=0.015, # Thermal leakage coefficient
        static_power_model="LINEAR",
    )
    power_estimator = PowerEstimator(config=power_config)
    print(f"  - Capacitance C_eff     : {power_config.capacitance * 1e9:.2f} nF")
    print(f"  - Base Static Power P0  : {power_config.base_static_power:.1f} W")
    print(f"  - Voltage Range         : [{power_config.min_voltage:.2f} V - {power_config.max_voltage:.2f} V]")
    print(f"  - Frequency Range       : [{power_config.min_frequency / 1e9:.1f} GHz - {power_config.max_frequency / 1e9:.1f} GHz]")

    # 3. Configure Thermal RC Model
    print("\n[Step 3] Configuring Thermal RC Model Subsystem...")
    thermal_config = ThermalConfig(
        ambient_temperature=25.0,     # 25.0 °C ambient
        initial_temperature=25.0,     # 25.0 °C initial junction temperature
        thermal_resistance=1.5,       # 1.5 °C/W thermal resistance
        thermal_capacitance=10.0,     # 10.0 J/°C thermal mass (tau = 15.0s)
    )
    thermal_model = ThermalModel(config=thermal_config)
    print(f"  - Ambient Temp T_amb    : {thermal_model.ambient_temperature:.1f} °C")
    print(f"  - Initial Temp T_init   : {thermal_model.initial_temperature:.1f} °C")
    print(f"  - Thermal Resistance Rth: {thermal_model.thermal_resistance:.2f} °C/W")
    print(f"  - Thermal Mass Cth      : {thermal_model.thermal_capacitance:.1f} J/°C")
    print(f"  - Time Constant tau     : {thermal_model.time_constant:.1f} s")

    # 4. Configure Thermal Safety Limits
    print("\n[Step 4] Configuring Thermal Safety Thresholds...")
    safety_config = ThermalSafetyConfig(
        warning_temperature=35.0,     # 35.0 °C Warning threshold
        critical_temperature=45.0,    # 45.0 °C Critical threshold
        maximum_temperature=55.0,     # 55.0 °C Overheating limit
        hysteresis=1.0,               # 1.0 °C De-escalation hysteresis
    )
    safety_monitor = ThermalSafetyMonitor(config=safety_config)
    print(f"  - WARNING Threshold     : {safety_config.warning_temperature:.1f} °C")
    print(f"  - CRITICAL Threshold    : {safety_config.critical_temperature:.1f} °C")
    print(f"  - OVERHEATING Threshold : {safety_config.maximum_temperature:.1f} °C")

    # 5. Create Simulation Engine
    print("\n[Step 5] Initializing ThermoShift Simulation Engine & DVFS Mapper...")
    dvfs_mapper = LinearDVFSMapper(power_config=power_config)
    sim_config = SimulationConfig(
        duration=25.0,
        timestep=1.0,
        ambient_temperature=25.0,
        initial_temperature=25.0,
        integration_method="EXACT",
    )
    simulation = ThermoShiftSimulation(
        power_estimator=power_estimator,
        thermal_model=thermal_model,
        dvfs_mapper=dvfs_mapper,
        config=sim_config,
    )

    # 6. Run Simulation
    print("\n[Step 6] Running End-to-End Simulation Pipeline (0.0s -> 25.0s)...")
    results = simulation.run(workload_profile=workload_profile, duration=25.0, timestep=1.0)
    print(f"  - Total Timesteps Executed: {len(results.steps)}")

    # 7. Analyze Results
    print("\n[Step 7] Analyzing Simulation Results via SimulationAnalyzer & Safety Monitor...")
    analyzer = SimulationAnalyzer()
    summary = analyzer.analyze(results, safety_monitor=safety_monitor)
    safety_analysis = results.analyze_safety(safety_monitor)

    # 8. Display Summary
    print("\n[Step 8] Simulation Summary Statistics:")
    print("-" * 95)
    print(f"  Workload Metrics:")
    print(f"    * Peak Workload          : {summary.peak_workload * 100.0:.1f} %")
    print(f"    * Average Workload       : {summary.average_workload * 100.0:.1f} %")
    print(f"  Power Dissipation Metrics:")
    print(f"    * Peak Dynamic Power     : {summary.peak_dynamic_power:.4f} W")
    print(f"    * Peak Static Power      : {summary.peak_static_power:.4f} W")
    print(f"    * Peak Total Power       : {summary.peak_total_power:.4f} W")
    print(f"    * Average Power          : {summary.average_total_power:.4f} W")
    print(f"  Temperature Evolution:")
    print(f"    * Peak Temperature       : {summary.peak_temperature:.2f} °C (at t = {summary.time_of_peak_temperature:.1f} s)")
    print(f"    * Final Temperature      : {summary.final_temperature:.2f} °C")
    print(f"    * Average Temperature    : {summary.average_temperature:.2f} °C")
    print("-" * 95)

    # 9, 10, 11, 12. Generate Plots with Thermal Threshold Overlays
    output_dir = os.path.dirname(__file__)
    w_path = os.path.join(output_dir, "plot_workload.png")
    p_path = os.path.join(output_dir, "plot_power.png")
    t_path = os.path.join(output_dir, "plot_temperature.png")
    o_path = os.path.join(output_dir, "plot_overview.png")

    print("\n[Step 9-12] Generating and Exporting Plots...")
    plotter = SimulationPlotter(safety_config=safety_config)
    plotter.plot_workload(results, save_path=w_path)
    plotter.plot_power(results, save_path=p_path)
    plotter.plot_temperature(results, save_path=t_path)
    plotter.plot_overview(results, save_path=o_path)
    print(f"  - Workload Plot Saved    : {w_path}")
    print(f"  - Power Plot Saved       : {p_path}")
    print(f"  - Temperature Plot Saved : {t_path}")
    print(f"  - Overview Dashboard Saved: {o_path}")

    # 13. Show Thermal Safety Information
    print("\n[Step 13] Thermal Safety Analysis & Boundary Transitions:")
    print("-" * 95)
    print(f"  Violations Occurred      : {safety_analysis.has_violations}")
    print(f"  Overheating Occurred     : {safety_analysis.has_overheating}")
    print(f"  State Durations:")
    print(f"    * NORMAL State Duration   : {safety_analysis.normal_duration:.1f} s ({safety_analysis.normal_timesteps} steps)")
    print(f"    * WARNING State Duration  : {safety_analysis.warning_duration:.1f} s ({safety_analysis.warning_timesteps} steps)")
    print(f"    * CRITICAL State Duration : {safety_analysis.critical_duration:.1f} s ({safety_analysis.critical_timesteps} steps)")
    print(f"    * OVERHEATING Duration    : {safety_analysis.overheating_duration:.1f} s ({safety_analysis.overheating_timesteps} steps)")
    print(f"  First Threshold Crossings:")
    print(f"    * First WARNING Time      : {safety_analysis.first_warning_time} s")
    print(f"    * First CRITICAL Time     : {safety_analysis.first_critical_time} s")
    print(f"    * First OVERHEATING Time  : {safety_analysis.first_overheating_time} s")

    print(f"\n  Detected State Transition Events ({len(safety_analysis.transitions)} events):")
    print("  " + "-" * 85)
    print(f"  {'Step':<5} | {'Time (s)':<8} | {'Temp (°C)':<9} | {'Previous State':<15} -> {'New State':<15} | {'Event Type'}")
    print("  " + "-" * 85)
    for event in safety_analysis.transitions:
        print(
            f"  #{event.step_index:<4} | "
            f"{event.time:<8.1f} | "
            f"{event.temperature:<9.2f} | "
            f"{event.previous_status.value:<15} -> "
            f"{event.new_status.value:<15} | "
            f"{event.event_type.value}"
        )
    print("  " + "-" * 85)

    print("=" * 95)
    print(" ThermoShift Master End-to-End Pipeline Demonstration Completed Successfully!")
    print("=" * 95)


if __name__ == "__main__":
    main()
