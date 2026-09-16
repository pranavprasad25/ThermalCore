"""ThermoShift Day 4 Task 1 — Per-Core Temperature Monitoring & History System Demonstration Script."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.temperature_monitor import TemperatureMonitor


def main() -> None:
    """Run demonstration of per-core temperature monitoring system."""
    print("=" * 80)
    print(" ThermoShift — Day 4 Task 1: Per-Core Temperature Monitoring & History Demo")
    print("=" * 80)

    # Instantiate monitor
    monitor = TemperatureMonitor(history_size=100)

    # 4 CPU Cores multi-step simulation readings dataset
    readings_timeline = [
        # Time 0
        (0.0, {0: 60.0, 1: 65.0, 2: 62.0, 3: 58.0}),
        # Time 1
        (1.0, {0: 61.0, 1: 68.0, 2: 64.0, 3: 59.0}),
        # Time 2
        (2.0, {0: 63.0, 1: 72.0, 2: 67.0, 3: 60.0}),
    ]

    for time_step, core_readings in readings_timeline:
        print(f"\n[Simulating Time {time_step:.1f}s Input Readings]")
        print("-" * 80)

        for core_id, temp in core_readings.items():
            monitor.update_temperature(core_id=core_id, temperature=temp, timestamp=time_step)

        # Display current status summary
        print(f"{'Core ID':<10} | {'Current (°C)':<14} | {'Previous (°C)':<14} | {'Delta T (°C)':<14} | {'Delta t (s)':<12} | {'History Len'}")
        print("-" * 80)

        for core_id in sorted(monitor.get_monitored_cores()):
            curr = monitor.get_current_temperature(core_id)
            prev = monitor.get_previous_temperature(core_id)
            delta = monitor.get_temperature_difference(core_id)
            dt = monitor.get_time_difference(core_id)
            hist_len = len(monitor.get_temperature_history(core_id))

            curr_str = f"{curr:.1f} °C" if curr is not None else "N/A"
            prev_str = f"{prev:.1f} °C" if prev is not None else "None"
            delta_str = f"{delta:+.1f} °C" if delta is not None else "None"
            dt_str = f"{dt:.1f} s" if dt is not None else "None"

            print(
                f"Core {core_id:<5} | "
                f"{curr_str:<14} | "
                f"{prev_str:<14} | "
                f"{delta_str:<14} | "
                f"{dt_str:<12} | "
                f"{hist_len}"
            )

    print("\n" + "=" * 80)
    print(" Detailed Core Spotlight State Summary:")
    print("=" * 80)

    for core_id in sorted(monitor.get_monitored_cores()):
        latest = monitor.get_latest_reading(core_id)
        if latest:
            print(f"\nCore {core_id}:")
            print(f"  Current Temperature    : {latest.current_temperature:.1f} °C")
            print(f"  Previous Temperature   : {latest.previous_temperature if latest.previous_temperature is not None else 'None'}")
            print(f"  Temperature Difference : {latest.temperature_difference if latest.temperature_difference is not None else 'None'}")
            print(f"  Time Difference        : {latest.time_difference if latest.time_difference is not None else 'None'}")
            print(f"  History Buffer         : {latest.history}")

    print("\n" + "=" * 80)
    print(" Temperature Monitoring & History System Demo Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
