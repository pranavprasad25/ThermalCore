"""ThermoShift Day 3 Task 4 — Thermal Safety Subsystem and Simulation Integration Demo."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.safety.config import ThermalSafetyConfig
from src.safety.monitor import ThermalSafetyMonitor
from src.safety.status import ThermalStatus
from src.simulation.config import SimulationConfig
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


def main() -> None:
    """Run runnable Thermal Safety Subsystem demonstration."""
    print("=" * 85)
    print(" ThermoShift — Thermal Safety Subsystem & Time-Series Analysis Demo")
    print("=" * 85)

    # 1. Configure Thermal Safety Monitor
    safety_config = ThermalSafetyConfig(
        warning_temperature=70.0,    # 70 °C Warning boundary
        critical_temperature=85.0,   # 85 °C Critical boundary
        maximum_temperature=95.0,    # 95 °C Maximum / Overheating boundary
        hysteresis=1.0,              # 1.0 °C hysteresis
    )
    monitor = ThermalSafetyMonitor(config=safety_config)

    print("\n[1] Configured Thermal Safety Limits:")
    print(f"    - Safe Range:               T < {safety_config.warning_temperature:.1f} °C  (NORMAL)")
    print(f"    - Warning Range:            {safety_config.warning_temperature:.1f} °C <= T < {safety_config.critical_temperature:.1f} °C  (WARNING)")
    print(f"    - Critical Range:           {safety_config.critical_temperature:.1f} °C <= T < {safety_config.maximum_temperature:.1f} °C  (CRITICAL)")
    print(f"    - Overheating Boundary:     T >= {safety_config.maximum_temperature:.1f} °C  (OVERHEATING)")
    print(f"    - De-escalation Hysteresis: {safety_config.hysteresis:.1f} °C")

    # 2. Check Single Temperatures Across All States
    sample_temps = [55.0, 72.5, 88.0, 97.5]
    print("\n[2] Single-Temperature Safety Checks:")
    print("-" * 85)
    print(f"{'Temp (°C)':<12} | {'Thermal Status':<16} | {'is_safe':<10} | {'is_warning':<12} | {'is_critical':<12} | {'is_overheating':<14}")
    print("-" * 85)
    for temp in sample_temps:
        res = monitor.check_temperature(temp)
        print(
            f"{res.temperature:<12.1f} | {res.status.value:<16} | {str(res.is_safe):<10} | "
            f"{str(res.is_warning):<12} | {str(res.is_critical):<12} | {str(res.is_overheating):<14}"
        )
    print("-" * 85)

    # 3. Initialize End-to-End Simulation Pipeline
    # High power configuration to trigger warning, critical, and overheating limits
    power_config = PowerConfig(
        capacitance=3.0e-9,           # 3.0 nF high package capacity
        base_static_power=12.0,       # 12 W base leakage
        min_voltage=0.8,
        max_voltage=1.45,             # 1.45 V max Turbo voltage
        min_frequency=1.0e9,
        max_frequency=4.0e9,          # 4.0 GHz max Turbo clock
        reference_temperature=25.0,
        temperature_coefficient=0.02, # Thermal leakage scaling
        static_power_model="EXPONENTIAL",
    )
    power_estimator = PowerEstimator(config=power_config)

    thermal_config = ThermalConfig(
        ambient_temperature=25.0,
        initial_temperature=25.0,
        thermal_resistance=2.2,       # 2.2 °C/W -> enables high power to reach CRITICAL and OVERHEATING states
        thermal_capacitance=12.0,     # tau = 26.4s
    )
    thermal_model = ThermalModel(config=thermal_config)
    dvfs_mapper = LinearDVFSMapper(power_config=power_config)

    simulation = ThermoShiftSimulation(
        power_estimator=power_estimator,
        thermal_model=thermal_model,
        dvfs_mapper=dvfs_mapper,
    )

    # 4. Define Dynamic Workload Profile (Idle -> Heavy Burst -> Cooldown / Recovery)
    timeline = [
        (0.0, 0.20),   # Idle
        (5.0, 1.00),   # Heavy Compute Burst (heats system)
        (20.0, 0.00),  # Idle Recovery (cools system down)
    ]
    profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

    print("\n[3] Running Simulation Engine with Thermal Safety Integration (0.0s -> 50.0s, dt=1.0s)...")
    sim_results = simulation.run(workload_profile=profile, duration=50.0, timestep=1.0)

    # 5. Perform Safety Analysis on Simulation Time Series
    analysis = monitor.analyze(sim_results)

    print("\n[4] Thermal Safety Time-Series Summary:")
    summary = analysis.summary()
    print(f"    - Maximum Temperature:         {summary['maximum_temperature']:.2f} °C (at t = {summary['time_of_maximum_temperature']:.1f}s)")
    print(f"    - Final Temperature:           {summary['final_temperature']:.2f} °C")
    print(f"    - Violations Occurred:          {summary['has_violations']}")
    print(f"    - Overheating Occurred:         {summary['has_overheating']}")
    print(f"    - First Warning Time:          {summary['first_warning_time']} s" if summary['first_warning_time'] is not None else "    - First Warning Time:          None")
    print(f"    - First Critical Time:         {summary['first_critical_time']} s" if summary['first_critical_time'] is not None else "    - First Critical Time:         None")
    print(f"    - First Overheating Time:      {summary['first_overheating_time']} s" if summary['first_overheating_time'] is not None else "    - First Overheating Time:      None")

    print("\n[5] State Duration Breakdown:")
    print(f"    - NORMAL State:                {summary['normal_timesteps']:<3} timesteps ({summary['normal_duration_seconds']:.1f} s)")
    print(f"    - WARNING State:               {summary['warning_timesteps']:<3} timesteps ({summary['warning_duration_seconds']:.1f} s)")
    print(f"    - CRITICAL State:              {summary['critical_timesteps']:<3} timesteps ({summary['critical_duration_seconds']:.1f} s)")
    print(f"    - OVERHEATING State:           {summary['overheating_timesteps']:<3} timesteps ({summary['overheating_duration_seconds']:.1f} s)")

    # 6. Display State Transition Events
    print("\n[6] Threshold Transition Events Log:")
    print("-" * 85)
    print(f"{'Time (s)':<10} | {'Temp (°C)':<12} | {'Prior Status':<16} | {'New Status':<16} | {'Event Type':<22}")
    print("-" * 85)
    for event in analysis.transitions:
        print(
            f"{event.time:<10.1f} | {event.temperature:<12.2f} | {event.previous_status.value:<16} | "
            f"{event.new_status.value:<16} | {event.event_type.value:<22}"
        )
    print("-" * 85)
    print("=" * 85)
    print(" Thermal Safety demonstration completed successfully!")
    print("=" * 85)


if __name__ == "__main__":
    main()
