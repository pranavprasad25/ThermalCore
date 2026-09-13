"""Demonstration script showing Thermal Data Processing & History in ThermoShift / ThermalCore."""

import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.config import ThermalConfig
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading
from src.thermal.sensor import SimulatedThermalSensor



def main() -> None:
    print("=" * 75)
    print(" ThermoShift / ThermalCore — Day 2 Task 2: Data Processing & History Demo")
    print("=" * 75)

    # 1. Instantiate Configuration, Acquisition, and Processing Pipeline
    config = ThermalConfig(
        history_capacity=20,
        moving_average_window=4,
        smoothing_enabled=True,
        smoothing_alpha=0.3,
        trend_tolerance=0.05,
    )

    acq = ThermalDataAcquisition(config=config)
    processor = ThermalDataProcessor(config=config)

    # 2. Simulate Core 0 Heating Run
    sensor_cpu0 = SimulatedThermalSensor(sensor_id="core_0", start_temp=40.0, noise_std=1.2, seed=10)
    sensor_cpu0.set_target_temperature(85.0)

    print("\n1. Processing Continuous Thermal Readings for Core 0 (Heating Workload):")
    print("-" * 75)
    print(f"{'Sample':<8} | {'Raw Temp':<10} | {'Smoothed Temp':<14} | {'Moving Avg':<12} | {'Rate (C/s)':<12} | {'Trend'}")

    print("-" * 75)

    from datetime import datetime, timezone, timedelta
    base_ts = datetime(2026, 9, 13, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(1, 11):
        raw_reading = acq.read_from_sensor(sensor_cpu0)
        if raw_reading:
            # Attach 1-second incremented timestamps for clear rate demonstration
            timestamped_reading = ThermalReading(
                temperature=raw_reading.temperature,
                timestamp=base_ts + timedelta(seconds=i),
                sensor_id=raw_reading.sensor_id,
            )
            processed = processor.process(timestamped_reading)
            print(
                f" #{i:<6} | "
                f"{processed.raw_reading.temperature:8.2f} C | "
                f"{processed.processed_temperature:12.2f} C | "
                f"{processed.moving_average:10.2f} C | "
                f"{processed.rate_of_change:10.2f} | "
                f"{processed.trend.value}"
            )


    # 3. Core 0 Statistics Overview
    print("\n2. Core 0 Statistical Summary:")
    print("-" * 75)
    print(f"  Current Temperature  : {processor.get_current_temperature('core_0'):.2f} C")
    print(f"  Minimum Temperature  : {processor.get_min_temperature('core_0'):.2f} C")
    print(f"  Maximum Temperature  : {processor.get_max_temperature('core_0'):.2f} C")
    print(f"  Average Temperature  : {processor.get_average_temperature('core_0'):.2f} C")
    print(f"  Moving Avg (Window=4): {processor.get_moving_average('core_0', window=4):.2f} C")
    print(f"  Rate of Change       : {processor.get_rate_of_change('core_0'):.2f} C/sec")
    print(f"  Thermal Trend        : {processor.get_trend('core_0').value}")

    # 4. Multi-Sensor Isolation Verification (Core 1 Cooling Run)
    sensor_cpu1 = SimulatedThermalSensor(sensor_id="core_1", start_temp=90.0, noise_std=0.5, seed=20)
    sensor_cpu1.set_target_temperature(45.0)

    for idx in range(1, 7):
        r = acq.read_from_sensor(sensor_cpu1)
        if r:
            ts_r = ThermalReading(
                temperature=r.temperature,
                timestamp=base_ts + timedelta(seconds=idx),
                sensor_id=r.sensor_id,
            )
            processor.process(ts_r)


    print("\n3. Multi-Sensor Isolation Check (Core 0 vs Core 1):")
    print("-" * 75)
    print(f"  Core 0 -> Avg Temp: {processor.get_average_temperature('core_0'):.2f} C | Trend: {processor.get_trend('core_0').value}")
    print(f"  Core 1 -> Avg Temp: {processor.get_average_temperature('core_1'):.2f} C | Trend: {processor.get_trend('core_1').value}")

    print("=" * 75)


if __name__ == "__main__":
    main()
