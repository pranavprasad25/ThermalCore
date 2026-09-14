"""Demonstration of Day 3 — Task 1: CPU Power Estimation in ThermoShift."""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.power.config import PowerConfig
from src.power.estimator import PowerEstimator


def print_separator(title: str = "") -> None:
    """Helper to print visual section separators."""
    if title:
        print(f"\n{'=' * 25} {title} {'=' * 25}")
    else:
        print("-" * 70)


def main() -> None:
    print("ThermoShift Power Estimation Subsystem Demo")
    print("=" * 70)

    # 1. Initialize Configuration and Power Estimator
    config = PowerConfig(
        capacitance=1.5e-9,  # 1.5 nF effective switching capacitance
        base_static_power=5.0,  # 5.0 W baseline static leakage
        activity_factor_scale=1.0,  # alpha = 1.0 * workload
        base_activity_factor=0.0,
        min_voltage=0.5,
        max_voltage=2.0,
        min_frequency=5.0e8,  # 500 MHz
        max_frequency=6.0e9,  # 6.0 GHz
        reference_temperature=25.0,
        temperature_coefficient=0.015,  # 1.5% leakage increase per deg C
        static_power_model="LINEAR",
    )
    estimator = PowerEstimator(config=config)

    print("Configuration Initialized:")
    print(f"  * Effective Capacitance (C): {config.capacitance * 1e9:.2f} nF")
    print(f"  * Base Static Power:         {config.base_static_power:.2f} W")
    print(f"  * Leakage Thermal Model:     {config.static_power_model} (coeff: {config.temperature_coefficient:.3f}/deg C)")

    # 2. Estimate Single Operating Point (as specified in Task Objective)
    print_separator("1. Single Operating Point Estimation")
    workload = 0.80  # 80% Utilization
    voltage = 1.10  # 1.10 V
    frequency = 3.0e9  # 3.00 GHz
    temperature = 65.0  # 65.0 deg C

    result = estimator.estimate(
        workload=workload,
        voltage=voltage,
        frequency=frequency,
        temperature=temperature,
        metadata={"core_id": "CPU_Core_0", "governor": "performance"},
    )

    print("Operating Conditions:")
    print(f"  * Workload / Utilization:    {result.workload * 100:.1f}%")
    print(f"  * Clock Frequency:           {result.frequency / 1e9:.2f} GHz ({result.frequency:,.0f} Hz)")
    print(f"  * Core Voltage:              {result.voltage:.2f} V")
    print(f"  * Core Temperature:          {result.temperature:.1f} deg C")
    print()
    print("Power Estimation Results:")
    print(f"  -> Dynamic Power (P_dyn):    {result.dynamic_power:.4f} W  [P = alpha * C * V^2 * f]")
    print(f"  -> Static Power  (P_stat):   {result.static_power:.4f} W  [P = P_base * (1 + beta * dT)]")
    print(f"  -> Total Power   (P_total):  {result.total_power:.4f} W  [P_total = P_dyn + P_stat]")

    # 3. Workload Scaling Effect (Idle to 100% Load)
    print_separator("2. Workload Scaling Effect (Fixed V = 1.1V, f = 3.0 GHz)")
    print(f"{'Workload (%)':<15}{'Dynamic (W)':<15}{'Static (W)':<15}{'Total (W)':<15}{'Dyn / Total (%)':<15}")
    print("-" * 75)

    workload_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for w in workload_steps:
        res = estimator.estimate(workload=w, voltage=1.1, frequency=3.0e9, temperature=65.0)
        dyn_pct = (res.dynamic_power / res.total_power) * 100 if res.total_power > 0 else 0.0
        print(f"{w * 100:>8.0f}%      {res.dynamic_power:>10.4f} W    {res.static_power:>10.4f} W    {res.total_power:>10.4f} W    {dyn_pct:>10.1f}%")

    # 4. DVFS (Dynamic Voltage and Frequency Scaling) Points
    print_separator("3. DVFS Operating Points (Workload = 80%)")
    print(f"{'State / Profile':<18}{'Voltage (V)':<14}{'Freq (GHz)':<14}{'Dynamic (W)':<14}{'Total (W)':<14}")
    print("-" * 74)

    dvfs_profiles = [
        ("Power Saver", 0.75, 1.2e9),
        ("Balanced Low", 0.90, 2.0e9),
        ("Nominal", 1.05, 2.8e9),
        ("Performance", 1.15, 3.4e9),
        ("Turbo Boost", 1.25, 4.0e9),
    ]

    for profile_name, v, f in dvfs_profiles:
        res = estimator.estimate(workload=0.80, voltage=v, frequency=f, temperature=65.0)
        print(f"{profile_name:<18}{res.voltage:>8.2f} V     {res.frequency / 1e9:>8.2f} GHz    {res.dynamic_power:>9.4f} W    {res.total_power:>9.4f} W")

    # 5. Temperature Impact on Leakage Power
    print_separator("4. Temperature Impact on Static Power (Leakage)")
    print(f"{'Temperature (deg C)':<22}{'Static Power (W)':<20}{'Static Growth (%)':<20}")
    print("-" * 62)

    temp_points = [25.0, 45.0, 65.0, 85.0, 105.0]
    base_leak = estimator.estimate_static_power(temperature=25.0)
    for t in temp_points:
        p_stat = estimator.estimate_static_power(temperature=t)
        growth = ((p_stat - base_leak) / base_leak) * 100
        print(f"{t:>12.1f} deg C         {p_stat:>10.4f} W           {growth:>+9.1f}%")

    print_separator("Power Estimation Subsystem Demo Completed Successfully")


if __name__ == "__main__":
    main()
