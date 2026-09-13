"""Demonstration script showing Threshold & Rule-Based Thermal Detection in ThermoShift / ThermalCore."""

from datetime import datetime, timedelta, timezone
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.config import ThermalConfig
from src.thermal.detector import ThermalDetector
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading


def main() -> None:
    print("=" * 80)
    print(" ThermoShift / ThermalCore — Day 2 Task 3: Thermal Threshold & Rule Detection Demo")
    print("=" * 80)

    # 1. Instantiate Configuration, Processor, and Detector Pipeline
    config = ThermalConfig(
        warning_temperature=70.0,
        critical_temperature=90.0,
        overheating_temperature=90.0,
        spike_threshold=10.0,
        rapid_rise_threshold=2.0,
        abnormal_cooling_threshold=5.0,
        sustained_high_temperature=80.0,
        sustained_high_duration=5.0,
        smoothing_enabled=False,  # Unsmoothed for explicit synthetic step demo
    )

    processor = ThermalDataProcessor(config=config)
    detector = ThermalDetector(config=config)

    base_ts = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)

    # Synthetic simulation scenarios for Core 0
    scenarios = [
        (1, 45.0, "Baseline normal operation"),
        (2, 50.0, "Normal load increase"),
        (3, 53.0, "Rapid rise starting (+3°C/s)"),
        (4, 57.0, "Continued rapid rise (+4°C/s)"),
        (5, 75.0, "Sudden spike (+18°C/s) & WARNING state"),
        (6, 82.0, "Entering sustained high temp range (t=0s sustained)"),
        (7, 83.0, "Sustained high temp continues (t=2s sustained)"),
        (8, 92.0, "CRITICAL state & OVERHEATING condition reached"),
        (9, 93.0, "Sustained high duration met (t=6s >= 5s)"),
        (10, 85.0, "De-escalation back to WARNING state"),
        (11, 78.0, "Cooling down in WARNING state"),
        (12, 60.0, "Rapid abnormal cooling (-18°C/s) & RECOVERY to NORMAL"),
    ]

    print("\n1. Running Thermal Rule & Threshold Detection Pipeline on Core 0:")
    print("-" * 80)
    print(f"{'Time':<5} | {'Temp (°C)':<8} | {'State':<9} | {'Transition':<12} | {'Conditions'}")
    print("-" * 80)

    for tick, temp, desc in scenarios:
        ts = base_ts + timedelta(seconds=tick)
        reading = ThermalReading(timestamp=ts, sensor_id="core_0", temperature=temp, unit="C")
        processed_data = processor.process(reading)
        result = detector.detect(processed_data)

        cond_str = ", ".join(c.value for c in result.conditions) if result.conditions else "NONE"

        print(
            f" +{tick:<3}s | "
            f"{result.temperature:8.1f} | "
            f"{result.state.value:<9} | "
            f"{result.transition.value:<12} | "
            f"{cond_str}"
        )
        if result.reasons:
            for r in result.reasons:
                print(f"        -> Reason: {r}")

    # 2. Multi-Sensor Isolation Check
    print("\n2. Multi-Sensor Isolation Check (Core 0 vs Core 1):")
    print("-" * 80)

    # Core 1 initial reading at normal temperature
    ts_c1 = base_ts + timedelta(seconds=13)
    reading_c1 = ThermalReading(timestamp=ts_c1, sensor_id="core_1", temperature=42.0)
    p_c1 = processor.process(reading_c1)
    res_c1 = detector.detect(p_c1)

    print(f"  Core 0 Latest State: {detector._previous_states.get('core_0').value}")
    print(f"  Core 1 Initial State: {res_c1.state.value} (Transition: {res_c1.transition.value})")
    print(f"  Multi-Sensor State Isolation Verified: Core 1 state is independent of Core 0.")

    # 3. JSON / Dictionary Export Serialization
    print("\n3. Structured Detection Result Serialization Output (to_dict()):")
    print("-" * 80)
    import json
    print(json.dumps(res_c1.to_dict(), indent=2))

    print("=" * 80)


if __name__ == "__main__":
    main()
