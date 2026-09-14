"""ThermoShift Day 3 Task 3 — End-to-End Workload -> Power -> Temperature Simulation Demo."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cpu.dvfs import LinearDVFSMapper
from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.simulation.config import SimulationConfig
from src.simulation.engine import ThermoShiftSimulation
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel
from src.workload.profile import WorkloadProfile


def main() -> None:
    """Run runnable end-to-end simulation demonstration."""
    print("=" * 80)
    print(" ThermoShift — End-to-End Workload -> Power -> Temperature Simulation")
    print("=" * 80)

    # 1. Define a realistic time-varying workload profile
    # Timeline: 0s->20% (idle), 2s->60% (startup), 5s->100% (burst), 10s->40% (throt), 15s->80% (steady), 20s->20% (cooldown)
    timeline = [
        (0.0, 0.20),
        (2.0, 0.60),
        (5.0, 1.00),
        (10.0, 0.40),
        (15.0, 0.80),
        (20.0, 0.20),
    ]
    workload_profile = WorkloadProfile.from_step(steps=timeline, interpolation="step")

    print(f"\n[1] Workload Profile Configured:")
    print(f"    - Duration: {workload_profile.duration:.1f} s")
    print(f"    - Peak Workload: {workload_profile.max_workload * 100:.1f} %")
    print(f"    - Timeline Points: {timeline}")

    # 2. Initialize Power Estimator
    power_config = PowerConfig(
        capacitance=1.5e-9,           # 1.5 nF
        base_static_power=4.5,        # 4.5 W baseline static leakage
        min_voltage=0.8,              # 0.8 V
        max_voltage=1.25,             # 1.25 V
        min_frequency=1.0e9,          # 1.0 GHz
        max_frequency=3.5e9,          # 3.5 GHz
        reference_temperature=25.0,   # 25 °C reference
        temperature_coefficient=0.015, # Thermal leakage scaling factor
        static_power_model="LINEAR",
    )
    power_estimator = PowerEstimator(config=power_config)

    # 3. Initialize Thermal Model
    thermal_config = ThermalConfig(
        ambient_temperature=25.0,     # 25 °C ambient
        initial_temperature=25.0,     # Start at ambient 25 °C
        thermal_resistance=1.2,       # 1.2 °C/W package thermal resistance
        thermal_capacitance=15.0,     # 15 J/°C thermal mass (tau = 18s)
    )
    thermal_model = ThermalModel(config=thermal_config)

    # 4. Initialize DVFS Mapper & Simulation Engine
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

    # 5. Execute Simulation Pipeline
    print("\n[2] Running Simulation Engine (0.0s -> 25.0s, dt=1.0s)...")
    results = simulation.run(workload_profile=workload_profile, duration=25.0)

    # 6. Display Timestep Log Table
    print("\n[3] Timestep Results Table:")
    print("-" * 88)
    print(
        f"{'Time (s)':<10} | {'Workload':<10} | {'Freq (GHz)':<12} | {'Voltage (V)':<12} | {'Total Power (W)':<16} | {'Temp (°C)':<10}"
    )
    print("-" * 88)

    for step in results:
        print(
            f"{step.time:<10.1f} | {step.workload * 100:<9.1f}% | {step.frequency / 1.0e9:<12.3f} | "
            f"{step.voltage:<12.3f} | {step.total_power:<16.4f} | {step.temperature:<10.2f}"
        )

    print("-" * 88)

    # 7. Display Summary Metrics
    summary = results.summary()
    print("\n[4] Simulation Summary Metrics:")
    print(f"    - Peak Workload:               {summary.peak_workload * 100:.1f} %")
    print(f"    - Peak Dynamic Power:          {summary.peak_dynamic_power:.4f} W")
    print(f"    - Peak Static Power:           {summary.peak_static_power:.4f} W")
    print(f"    - Peak Total Power:            {summary.peak_total_power:.4f} W  (at t = {summary.time_at_peak_power:.1f}s)")
    print(f"    - Average Total Power:         {summary.average_power:.4f} W")
    print(f"    - Peak Temperature:            {summary.peak_temperature:.2f} °C  (at t = {summary.time_at_peak_temperature:.1f}s)")
    print(f"    - Final Temperature:           {summary.final_temperature:.2f} °C")
    print(f"    - Average Temperature:         {summary.average_temperature:.2f} °C")
    print(f"    - Final Steady-State Estimate: {summary.steady_state_final_estimate:.2f} °C")
    print("=" * 80)
    print(" Simulation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
