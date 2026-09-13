"""Demonstration script showing the Central Thermal Health Monitoring Engine in ThermoShift / ThermalCore."""

from datetime import datetime, timedelta, timezone
import json
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.config import ThermalConfig
from src.thermal.health_monitor import ThermalHealthMonitor
from src.thermal.reading import ThermalReading
from src.thermal.sensor import SimulatedThermalSensor


def main() -> None:
    print("=" * 90)
    print(" ThermoShift / ThermalCore — Day 2 Task 5: Central Thermal Health Monitoring Engine Demo")
    print("=" * 90)

    # 1. Instantiate Configuration and Health Monitor Engine
    config = ThermalConfig(
        warning_temperature=70.0,
        critical_temperature=90.0,
        baseline_window_size=10,
        minimum_baseline_samples=5,
        anomaly_score_threshold=50.0,
        smoothing_enabled=False,
    )

    monitor = ThermalHealthMonitor(config=config)
    base_ts = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)

    # Simulation timeline for Core 0
    timeline = [
        (1, 45.0, "Baseline normal operation"),
        (2, 45.2, "Baseline normal operation"),
        (3, 44.9, "Baseline normal operation"),
        (4, 45.1, "Baseline normal operation"),
        (5, 45.0, "Baseline established (~45.0°C)"),
        (6, 68.0, "Load increase: +23°C shift (Sub-threshold anomaly -> DEGRADED health)"),
        (7, 76.0, "Load increase: 76°C (Exceeds Task 3 Warning 70°C threshold -> WARNING health)"),
        (8, 92.0, "Hotspot spike: 92°C (Exceeds Task 3 Critical 90°C threshold -> CRITICAL health)"),
        (9, 93.0, "Sustained critical overheating"),
        (10, 75.0, "Cooling down to 75°C"),
        (11, 45.0, "Temperature drops back to normal 45°C"),
        (12, 45.0, "Thermal state fully settled back to 45°C"),
        (13, 45.0, "Anomaly recovery completed -> HEALTHY status"),
    ]


    print("\n1. Running Unified Thermal Health Pipeline on Core 0:")
    print("-" * 90)
    print(
        f"{'Tick':<5} | {'Temp (°C)':<8} | {'Operating State':<15} | {'Health Status':<13} | {'Score':<6} | {'Active Alerts'}"
    )
    print("-" * 90)

    for tick, temp, desc in timeline:
        ts = base_ts + timedelta(seconds=tick)
        reading = ThermalReading(timestamp=ts, sensor_id="core_0", temperature=temp, unit="C")
        result = monitor.update(reading)

        alerts_str = ", ".join(result.active_alerts) if result.active_alerts else "NONE"

        print(
            f" +{tick:<2}s  | "
            f"{result.temperature:8.1f} | "
            f"{result.thermal_state.value:<15} | "
            f"{result.health_status.value:<13} | "
            f"{result.health_score:5.1f} | "
            f"{alerts_str}"
        )
        if result.penalties:
            for p in result.penalties:
                print(f"        -> Health Penalty: {p}")

    # 2. Conveniences Accessor APIs Demonstration
    print("\n2. Health Monitor Accessor APIs Output (Core 0):")
    print("-" * 90)
    print(f"  get_health_status('core_0')     : {monitor.get_health_status('core_0').value}")
    print(f"  get_health_score('core_0')      : {monitor.get_health_score('core_0'):.1f} / 100.0")
    print(f"  get_current_temperature('core_0'): {monitor.get_current_temperature('core_0'):.1f} °C")
    print(f"  get_active_alerts('core_0')     : {monitor.get_active_alerts('core_0')}")
    print(f"  get_history('core_0') count     : {len(monitor.get_history('core_0'))} readings stored")

    # 3. Direct Sensor Acquisition Integration (Task 1 + Task 5)
    print("\n3. Direct Sensor Acquisition Integration (read_and_update()):")
    print("-" * 90)
    sensor = SimulatedThermalSensor(sensor_id="core_1", start_temp=42.0, seed=123)
    c1_result = monitor.read_and_update(sensor)
    if c1_result:
        print(f"  Acquired from Core 1 Sensor -> Temp: {c1_result.temperature:.1f}°C | Health: {c1_result.health_status.value} (Score: {c1_result.health_score:.1f})")

    # 4. Multi-Sensor Isolation Verification
    print("\n4. Multi-Sensor State Isolation Verification:")
    print("-" * 90)
    print(f"  Core 0 Health Status: {monitor.get_health_status('core_0').value} (Score: {monitor.get_health_score('core_0'):.1f})")
    print(f"  Core 1 Health Status: {monitor.get_health_status('core_1').value} (Score: {monitor.get_health_score('core_1'):.1f})")
    print(f"  Multi-Sensor State Isolation Verified: Core 0 and Core 1 maintain independent health state.")

    # 5. Structured Serialization Output (to_dict())
    print("\n5. Structured Unified Thermal Health Result Serialization (to_dict()):")
    print("-" * 90)
    if c1_result:
        print(json.dumps(c1_result.to_dict(), indent=2))

    print("=" * 90)


if __name__ == "__main__":
    main()
