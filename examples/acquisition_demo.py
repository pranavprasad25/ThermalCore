"""Demonstration script showing Thermal Data Acquisition pipeline in ThermoShift / ThermalCore."""

import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidReadingError
from src.thermal.sensor import SimulatedThermalSensor


def main() -> None:
    print("=" * 75)
    print(" ThermoShift / ThermalCore — Day 2 Task 1: Data Acquisition Demo")
    print("=" * 75)

    # 1. Instantiate Configuration and Acquisition Pipeline
    config = ThermalConfig(min_valid_temp=-40.0, max_valid_temp=125.0)
    acq = ThermalDataAcquisition(config=config, buffer_capacity=10)

    print("\n1. Direct Ingestion & Unit Normalization:")
    print("-" * 75)
    r1 = acq.ingest_reading(temperature=25.0, sensor_id="core_sensor_0", unit="C")
    r2 = acq.ingest_reading(temperature=98.6, sensor_id="core_sensor_1", unit="F")
    r3 = acq.ingest_reading(temperature=310.15, sensor_id="core_sensor_2", unit="K")

    print(f"  Ingested 25.0 C -> {r1.temperature:.2f} C (Sensor: {r1.sensor_id})")
    print(f"  Ingested 98.6 F -> {r2.temperature:.2f} C (Sensor: {r2.sensor_id})")
    print(f"  Ingested 310.15 K -> {r3.temperature:.2f} C (Sensor: {r3.sensor_id})")

    # 2. Rejection of Invalid Input Data
    print("\n2. Input Data Validation & Rejection:")
    print("-" * 75)
    invalid_inputs = [None, float("nan"), float("inf"), "invalid_str", True, 250.0]
    for inp in invalid_inputs:
        try:
            acq.ingest_reading(temperature=inp, sensor_id="test_sensor")
        except InvalidReadingError as err:
            print(f"  Rejected invalid input {inp!r:<15} -> {err}")

    # 3. Simulated Sensor Streaming
    print("\n3. Simulated Sensor Stream (Finite 5-Sample Stream):")
    print("-" * 75)
    sensor = SimulatedThermalSensor(
        sensor_id="sim_sensor_cpu0",
        start_temp=45.0,
        noise_std=0.8,
        seed=42,
    )
    sensor.set_target_temperature(75.0)

    for reading in acq.stream(sensor, count=5, interval=0.0):
        ts_str = reading.timestamp.strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts_str}] Sensor: {reading.sensor_id} | Temp: {reading.temperature:6.2f} C | Unit: {reading.unit}")

    # 4. Reading Buffer Inspection
    print("\n4. Recent Reading Buffer Contents:")
    print("-" * 75)
    buffered = acq.buffer.get_all()
    print(f"  Buffer size: {len(buffered)} / {acq.buffer.max_capacity}")
    for idx, r in enumerate(buffered, 1):
        print(f"   [{idx:02d}] Sensor: {r.sensor_id:<18} | Temp: {r.temperature:6.2f} C | Timestamp: {r.timestamp.isoformat()}")


    print("=" * 75)


if __name__ == "__main__":
    main()
