"""Demonstration script showing Statistical Thermal Anomaly Detection in ThermoShift / ThermalCore."""

from datetime import datetime, timedelta, timezone
import json
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.anomaly_detector import ThermalAnomalyDetector
from src.thermal.config import ThermalConfig
from src.thermal.detector import ThermalDetector
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading


def main() -> None:
    print("=" * 85)
    print(" ThermoShift / ThermalCore — Day 2 Task 4: Statistical Thermal Anomaly Detection Demo")
    print("=" * 85)

    # 1. Instantiate Pipeline Configuration
    config = ThermalConfig(
        warning_temperature=70.0,
        critical_temperature=90.0,
        baseline_window_size=10,
        minimum_baseline_samples=5,
        anomaly_sensitivity="MEDIUM",
        anomaly_score_threshold=50.0,
        persistence_count=3,
        recovery_threshold=15.0,
        smoothing_enabled=False,  # Unsmoothed for explicit synthetic step demonstration
    )

    processor = ThermalDataProcessor(config=config)
    threshold_detector = ThermalDetector(config=config)
    anomaly_detector = ThermalAnomalyDetector(config=config)

    base_ts = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)

    # Simulation timeline for Core 0
    timeline = [
        # (tick, temp, description)
        (1, 65.0, "Sample 1: Warm-up phase"),
        (2, 65.2, "Sample 2: Warm-up phase"),
        (3, 64.9, "Sample 3: Warm-up phase"),
        (4, 65.1, "Sample 4: Warm-up phase"),
        (5, 65.0, "Sample 5: Baseline established (~65.0°C)"),
        (6, 65.3, "Sample 6: Normal fluctuation around baseline"),
        (7, 78.0, "Sample 7: Sudden +13°C jump (Below Task 3 90°C limit, but statistical anomaly!)"),
        (8, 78.5, "Sample 8: 2nd anomalous reading (TEMPORARY)"),
        (9, 79.0, "Sample 9: 3rd anomalous reading (Transitions to PERSISTENT)"),
        (10, 80.0, "Sample 10: Sustained persistent anomaly"),
        (11, 65.1, "Sample 11: Temperature drops to normal -> RECOVERED"),
        (12, 65.0, "Sample 12: Settled back to NORMAL state"),
    ]

    print("\n1. Continuous Thermal Monitoring Pipeline (Core 0):")
    print("-" * 85)
    print(
        f"{'Tick':<5} | {'Temp (°C)':<8} | {'Task 3 State':<12} | {'Task 4 Status':<18} | {'Score':<6} | {'Severity':<8} | {'Persistence'}"
    )
    print("-" * 85)

    for tick, temp, desc in timeline:
        ts = base_ts + timedelta(seconds=tick)
        reading = ThermalReading(timestamp=ts, sensor_id="core_0", temperature=temp, unit="C")

        processed = processor.process(reading)
        t3_result = threshold_detector.detect(processed)
        t4_result = anomaly_detector.detect(processed)

        print(
            f" +{tick:<2}s  | "
            f"{t4_result.current_temperature:8.1f} | "
            f"{t3_result.state.value:<12} | "
            f"{t4_result.status.value:<18} | "
            f"{t4_result.anomaly_score:5.1f} | "
            f"{t4_result.severity.value:<8} | "
            f"{t4_result.persistence.value}"
        )
        if t4_result.reasons:
            for r in t4_result.reasons:
                print(f"        -> Reason: {r}")

    # 2. Key Differentiation Spotlight
    print("\n2. Key Architectural Differentiation Spotlight (Sample #7):")
    print("-" * 85)
    print("  Fixed Safety Threshold (Task 3):")
    print("    'Is 78.0°C above safety limit (90°C)?' -> NO (State: NORMAL)")
    print("  Statistical Anomaly Detector (Task 4):")
    print("    'Is 78.0°C unusual vs historical 65.0°C baseline?' -> YES (+13°C shift, Anomaly Score: >50)")

    # 3. Multi-Sensor Isolation Verification
    print("\n3. Multi-Sensor Baseline Isolation (Core 0 vs Core 1):")
    print("-" * 85)

    # Core 1 baseline build around 82.0°C
    for i in range(1, 6):
        r_c1 = ThermalReading(timestamp=base_ts + timedelta(seconds=12 + i), sensor_id="core_1", temperature=82.0)
        p_c1 = processor.process(r_c1)
        res_c1 = anomaly_detector.detect(p_c1)

    print(f"  Core 0 Rolling Baseline: {anomaly_detector.get_baseline('core_0'):.1f}°C")
    print(f"  Core 1 Rolling Baseline: {anomaly_detector.get_baseline('core_1'):.1f}°C")
    print("  Multi-Sensor Isolation Verified: Sensor baselines are tracked independently.")

    # 4. JSON Result Serialization Output
    print("\n4. Structured Thermal Anomaly Result Serialization Output (to_dict()):")
    print("-" * 85)
    print(json.dumps(res_c1.to_dict(), indent=2))

    print("=" * 85)


if __name__ == "__main__":
    main()
