"""Demonstration of Day 3 — Task 2: Thermal RC Modeling in ThermoShift."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator
from src.thermal.config import ThermalConfig
from src.thermal.thermal_model import ThermalModel


def print_separator(title: str = "") -> None:
    """Helper to print visual section separators."""
    if title:
        print(f"\n{'=' * 25} {title} {'=' * 25}")
    else:
        print("-" * 75)


def main() -> None:
    print("ThermoShift Thermal RC Model Subsystem Demo")
    print("=" * 75)

    # 1. Initialize Thermal Parameters and Model
    r_th = 1.2  # 1.2 °C/W
    c_th = 20.0  # 20.0 J/°C
    t_ambient = 25.0  # 25.0 °C
    t_initial = 25.0  # 25.0 °C

    model = ThermalModel(
        thermal_resistance=r_th,
        thermal_capacitance=c_th,
        ambient_temperature=t_ambient,
        initial_temperature=t_initial,
    )

    print("Thermal RC Model Initialized:")
    print(f"  * Thermal Resistance (R_th):    {model.thermal_resistance:.2f} deg C/W")
    print(f"  * Thermal Capacitance (C_th):   {model.thermal_capacitance:.2f} J/deg C")
    print(f"  * Thermal Time Constant (tau):  {model.time_constant:.2f} s [tau = R_th * C_th]")
    print(f"  * Ambient Temperature:          {model.ambient_temperature:.1f} deg C")
    print(f"  * Initial Temperature:          {model.initial_temperature:.1f} deg C")

    # 2. Steady-State Calculation
    print_separator("1. Steady-State Calculations")
    test_powers = [0.0, 10.0, 25.0, 45.0, 60.0]
    print(f"{'Power (W)':<15}{'Delta T (deg C)':<20}{'T_steady (deg C)':<20}{'Formula'}")
    print("-" * 75)
    for p in test_powers:
        delta_t = model.temperature_rise(power=p)
        t_steady = model.steady_state_temperature(power=p)
        print(f"{p:>8.1f} W      {delta_t:>12.2f} deg C     {t_steady:>12.2f} deg C       T_amb + P * R_th")

    # 3. Constant Power Transient Heating Simulation
    constant_power = 25.0  # Watts
    duration = 120.0  # seconds (5 * tau)
    timestep = 5.0  # seconds

    t_steady_expected = model.steady_state_temperature(constant_power)
    print_separator(f"2. Transient Heating Simulation (P = {constant_power:.1f} W, T_steady = {t_steady_expected:.2f} deg C)")
    print(f"{'Time (s)':<12}{'Power (W)':<14}{'Temp (deg C)':<16}{'Rise (deg C)':<16}{'Steady-State %':<15}")
    print("-" * 75)

    sim_res = model.simulate(power=constant_power, duration=duration, timestep=timestep)

    # Sample steps to print
    for step in sim_res.steps:
        # Calculate percentage of steady-state rise achieved
        rise_fraction = (step.temperature_rise / (constant_power * r_th)) * 100 if constant_power > 0 else 100.0
        print(f"{step.time:>6.1f} s     {step.power:>8.2f} W     {step.temperature:>10.2f} deg C    {step.temperature_rise:>10.2f} deg C    {rise_fraction:>9.1f}%")

    print(f"\nFinal Temperature Reached: {sim_res.final_temperature:.2f} deg C (Converged to Steady-State {t_steady_expected:.2f} deg C)")

    # 4. End-to-End Pipeline: Power Estimator -> Thermal Model with Dynamic DVFS Profile
    print_separator("3. End-to-End Power Estimator -> Thermal Model Integration")

    power_estimator = PowerEstimator(
        config=PowerConfig(
            capacitance=1.5e-9,
            base_static_power=5.0,
            temperature_coefficient=0.015,
            static_power_model="LINEAR",
        )
    )

    # Define multi-phase workload scenario
    phases = [
        ("Phase 1: Boot & Idle", 0.05, 0.80, 1.2e9, 15.0),  # 15s idle
        ("Phase 2: Heavy Load", 0.95, 1.20, 3.6e9, 35.0),  # 35s heavy compute
        ("Phase 3: Sustained Medium", 0.50, 1.05, 2.8e9, 30.0),  # 30s sustained
        ("Phase 4: Cooldown Idle", 0.00, 0.75, 1.0e9, 40.0),  # 40s cooling
    ]

    print("Simulating Dynamic Multi-Phase Workload:")
    model.reset(initial_temperature=25.0)

    total_time = 0.0
    dt_step = 1.0

    print(f"{'Phase Name':<28}{'Time (s)':<12}{'Workload':<12}{'Power (W)':<14}{'Temp (deg C)':<14}")
    print("-" * 80)

    for phase_name, workload, voltage, frequency, phase_duration in phases:
        n_steps = int(phase_duration / dt_step)
        for _ in range(n_steps):
            curr_temp = model.current_temperature
            power_res = power_estimator.estimate(
                workload=workload,
                voltage=voltage,
                frequency=frequency,
                temperature=curr_temp,
            )
            model.step(power=power_res, dt=dt_step)
            total_time += dt_step

        print(f"{phase_name:<28}{total_time:>6.1f} s     {workload * 100:>6.0f}%     {power_res.total_power:>8.2f} W     {model.current_temperature:>9.2f} deg C")

    print_separator("Thermal Model Subsystem Demo Completed Successfully")


if __name__ == "__main__":
    main()
