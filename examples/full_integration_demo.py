"""Full integration, thermal scenario simulation, and validation demo for ThermoShift / ThermalCore."""

from datetime import datetime, timezone
import json
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.config import ThermalConfig
from src.thermal.scenarios import get_all_standard_scenarios
from src.thermal.simulation import ThermalScenarioRunner


def main() -> None:
    print("=" * 105)
    print(" ThermoShift / ThermalCore — Day 2 Task 6: Thermal Simulation & Full Integration Validation Demo")
    print("=" * 105)

    # 1. Instantiate Configuration and Real Scenario Runner
    config = ThermalConfig(
        warning_temperature=70.0,
        critical_temperature=90.0,
        baseline_window_size=10,
        minimum_baseline_samples=5,
        anomaly_score_threshold=50.0,
        smoothing_enabled=False,
    )

    runner = ThermalScenarioRunner(config=config)
    scenarios = get_all_standard_scenarios()

    print("\n1. Executing 12 Standard Thermal Scenarios through REAL Task 1 -> Task 5 Pipeline:")
    print("-" * 105)
    print(
        f"{'#':<3} | {'Scenario Name':<32} | {'Samples':<7} | {'Temp Range (°C)':<16} | {'Max Severity':<12} | {'Min Health':<10} | {'Final Health':<12} | {'Result'}"
    )
    print("-" * 105)

    reports = []
    for idx, scenario in enumerate(scenarios, start=1):
        report = runner.run(scenario, seed=42)
        reports.append(report)

        pass_str = "PASS [OK]" if report.passed else "FAIL [X]"
        temp_range_str = f"{report.min_temperature:.1f} - {report.max_temperature:.1f}"

        print(
            f"#{idx:<2} | "
            f"{scenario.name:<32} | "
            f"{report.total_samples:<7} | "
            f"{temp_range_str:<16} | "
            f"{report.max_anomaly_severity:<12} | "
            f"{report.min_health_score:<10.1f} | "
            f"{report.final_health_status:<12} | "
            f"{pass_str}"
        )

    # 2. Scenarios Summary Statistics
    passed_count = sum(1 for r in reports if r.passed)
    total_count = len(reports)

    print("\n2. Simulation Suite Execution Summary:")
    print("-" * 105)
    print(f"  Total Scenarios Executed : {total_count}")
    print(f"  Passed Assertions        : {passed_count} / {total_count}")
    print(f"  Success Rate             : {(passed_count / total_count) * 100.0:.1f}%")
    print(f"  Target Pipeline Verified : Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 (Real Implementations)")

    # 3. Detailed Spotlight on Scenario 9 (Recovery Lifecycle)
    rec_report = reports[8]  # Scenario 9 (Recovery)
    print(f"\n3. Spotlight: Scenario 9 Step-by-Step Recovery Lifecycle Execution:")
    print("-" * 105)
    print(
        f"{'Step':<5} | {'Temp (°C)':<8} | {'Operating State':<15} | {'Anomaly Severity':<16} | {'Health Status':<13} | {'Health Score':<12} | {'Active Alerts'}"
    )
    print("-" * 105)

    for s in rec_report.steps[::3]:  # Print every 3rd step for scannability
        alerts_str = ", ".join(s.active_alerts) if s.active_alerts else "NONE"
        print(
            f" #{s.step_index:<3} | "
            f"{s.raw_temperature:8.1f} | "
            f"{s.thermal_state:<15} | "
            f"{s.anomaly_severity:<16} | "
            f"{s.health_status:<13} | "
            f"{s.health_score:12.1f} | "
            f"{alerts_str}"
        )

    # 4. Structured JSON Report Output
    print("\n4. Structured Scenario Report Serialization Output (to_dict()):")
    print("-" * 105)
    print(json.dumps(reports[0].to_dict(), indent=2))

    print("=" * 105)


if __name__ == "__main__":
    main()
