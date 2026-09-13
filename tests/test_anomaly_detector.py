"""Comprehensive automated test suite for ThermalAnomalyDetector."""

from datetime import datetime, timedelta, timezone
import math
from typing import Optional
import unittest


from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.anomaly_detector import ThermalAnomalyDetector
from src.thermal.anomaly_result import (
    AnomalyPersistence,
    AnomalySeverity,
    AnomalyStatus,
    AnomalyTransition,
    ThermalAnomalyResult,
)
from src.thermal.config import ThermalConfig
from src.thermal.detector import ThermalDetector
from src.thermal.exceptions import InvalidReadingError
from src.thermal.processed_data import ProcessedThermalData, ThermalTrend
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading
from src.thermal.sensor import SimulatedThermalSensor


class TestThermalAnomalyDetector(unittest.TestCase):
    """Unit test suite for statistical anomaly detection engine."""

    def setUp(self):
        """Set up standard configuration and detector instance for testing."""
        self.config = ThermalConfig(
            baseline_window_size=10,
            minimum_baseline_samples=5,
            anomaly_sensitivity="MEDIUM",
            anomaly_score_threshold=50.0,
            low_severity_threshold=20.0,
            medium_severity_threshold=50.0,
            high_severity_threshold=75.0,
            critical_severity_threshold=90.0,
            persistence_count=3,
            persistence_duration=3.0,
            recovery_threshold=15.0,
            minimum_variability=0.5,
            smoothing_enabled=False,
        )
        self.detector = ThermalAnomalyDetector(self.config)
        self.now = datetime.now(timezone.utc)

    def _create_processed_data(
        self,
        sensor_id: str = "core_0",
        temp: float = 65.0,
        raw_temp: Optional[float] = None,
        rate_of_change: float = 0.0,
        timestamp: datetime = None,
    ) -> ProcessedThermalData:
        """Helper factory to construct test ProcessedThermalData objects."""
        ts = timestamp if timestamp is not None else self.now
        actual_raw = raw_temp if raw_temp is not None else temp
        reading = ThermalReading(
            timestamp=ts,
            sensor_id=sensor_id,
            temperature=actual_raw,
            unit="C",
        )
        return ProcessedThermalData(
            raw_reading=reading,
            processed_temperature=temp,
            moving_average=temp,
            rate_of_change=rate_of_change,
            trend=ThermalTrend.STABLE,
        )


    # -------------------------------------------------------------------------
    # 1. Baseline Tests (1-7)
    # -------------------------------------------------------------------------
    def test_baseline_calculation(self):
        """Verify baseline calculation uses rolling median of readings."""
        for i, val in enumerate([64.0, 65.0, 66.0, 65.0, 64.0]):
            d = self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i))
            res = self.detector.detect(d)

        self.assertIsNotNone(res.baseline_temperature)
        self.assertEqual(res.baseline_temperature, 65.0)

    def test_rolling_baseline_window_limit(self):
        """Verify baseline window limits old readings and updates as new data arrives."""
        # Fill 10 readings at 60°C
        for i in range(10):
            d = self._create_processed_data(temp=60.0, timestamp=self.now + timedelta(seconds=i))
            self.detector.detect(d)

        # Shift next 10 readings to 80°C
        for i in range(10, 20):
            d = self._create_processed_data(temp=80.0, timestamp=self.now + timedelta(seconds=i))
            res = self.detector.detect(d)

        # Window capacity is 10, so baseline should roll completely to 80°C
        self.assertEqual(res.baseline_temperature, 80.0)

    def test_minimum_baseline_samples(self):
        """Verify INSUFFICIENT_DATA status until minimum samples threshold is reached."""
        for i in range(4):  # < 5 samples
            d = self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=i))
            res = self.detector.detect(d)
            self.assertEqual(res.status, AnomalyStatus.INSUFFICIENT_DATA)
            self.assertIsNone(res.baseline_temperature)

        # 5th reading meets minimum
        d5 = self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=4))
        res5 = self.detector.detect(d5)
        self.assertIsNotNone(res5.baseline_temperature)
        self.assertNotEqual(res5.status, AnomalyStatus.INSUFFICIENT_DATA)

    def test_spike_robustness(self):
        """Verify median baseline is robust against isolated spikes."""
        temps = [65.0, 65.0, 66.0, 65.0, 95.0, 65.0]  # Single 95°C spike
        for i, val in enumerate(temps):
            d = self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i))
            res = self.detector.detect(d)

        self.assertEqual(res.baseline_temperature, 65.0)

    # -------------------------------------------------------------------------
    # 2. Deviation & Statistical Tests (8-14)
    # -------------------------------------------------------------------------
    def test_positive_and_negative_deviation(self):
        """Verify signed deviation and absolute deviation calculations."""
        for i, val in enumerate([65.0, 65.0, 65.0, 65.0, 65.0]):
            self.detector.detect(self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i)))

        # Positive deviation (+10°C)
        d_pos = self._create_processed_data(temp=75.0, timestamp=self.now + timedelta(seconds=5))
        res_pos = self.detector.detect(d_pos)
        self.assertEqual(res_pos.deviation, 10.0)
        self.assertEqual(res_pos.absolute_deviation, 10.0)

        # Reset and test negative deviation (-10°C)
        self.detector.reset()
        for i, val in enumerate([65.0, 65.0, 65.0, 65.0, 65.0]):
            self.detector.detect(self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i)))

        d_neg = self._create_processed_data(temp=55.0, timestamp=self.now + timedelta(seconds=5))
        res_neg = self.detector.detect(d_neg)
        self.assertEqual(res_neg.deviation, -10.0)
        self.assertEqual(res_neg.absolute_deviation, 10.0)

    def test_zero_variance_handling(self):
        """Verify division by zero is safely prevented when variance is zero."""
        for i in range(6):
            d = self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=i))
            res = self.detector.detect(d)

        self.assertGreaterEqual(res.variability, self.config.minimum_variability)
        self.assertFalse(math.isnan(res.normalized_deviation))
        self.assertFalse(math.isinf(res.normalized_deviation))

    # -------------------------------------------------------------------------
    # 3. Sensitivity & Score Tests (15-23)
    # -------------------------------------------------------------------------
    def test_sensitivity_levels(self):
        """Verify LOW, MEDIUM, HIGH sensitivity configurations affect scores appropriately."""
        # LOW sensitivity
        cfg_low = ThermalConfig(baseline_window_size=10, minimum_baseline_samples=5, anomaly_sensitivity="LOW", smoothing_enabled=False)
        det_low = ThermalAnomalyDetector(cfg_low)
        for i, val in enumerate([64.8, 65.2, 65.0, 64.9, 65.1]):
            det_low.detect(self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i)))
        res_low = det_low.detect(self._create_processed_data(temp=67.5, timestamp=self.now + timedelta(seconds=5)))

        # HIGH sensitivity
        cfg_high = ThermalConfig(baseline_window_size=10, minimum_baseline_samples=5, anomaly_sensitivity="HIGH", smoothing_enabled=False)
        det_high = ThermalAnomalyDetector(cfg_high)
        for i, val in enumerate([64.8, 65.2, 65.0, 64.9, 65.1]):
            det_high.detect(self._create_processed_data(temp=val, timestamp=self.now + timedelta(seconds=i)))
        res_high = det_high.detect(self._create_processed_data(temp=67.5, timestamp=self.now + timedelta(seconds=5)))

        self.assertGreater(res_high.anomaly_score, res_low.anomaly_score)


    def test_score_bounded_range(self):
        """Verify anomaly score remains strictly within [0.0, 100.0]."""
        for i in range(5):
            self.detector.detect(self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=i)))

        # Extreme high jump
        res = self.detector.detect(self._create_processed_data(temp=145.0, timestamp=self.now + timedelta(seconds=5)))
        self.assertLessEqual(res.anomaly_score, 100.0)
        self.assertGreaterEqual(res.anomaly_score, 0.0)

    # -------------------------------------------------------------------------
    # 4. Severity Tests (24-29)
    # -------------------------------------------------------------------------
    def test_severity_levels(self):
        """Verify severity levels NONE, LOW, MEDIUM, HIGH, CRITICAL based on score."""
        self.assertEqual(self.detector.classify_severity(10.0), AnomalySeverity.NONE)
        self.assertEqual(self.detector.classify_severity(30.0), AnomalySeverity.LOW)
        self.assertEqual(self.detector.classify_severity(60.0), AnomalySeverity.MEDIUM)
        self.assertEqual(self.detector.classify_severity(80.0), AnomalySeverity.HIGH)
        self.assertEqual(self.detector.classify_severity(95.0), AnomalySeverity.CRITICAL)

    # -------------------------------------------------------------------------
    # 5. Persistence Tests (30-35)
    # -------------------------------------------------------------------------
    def test_temporary_vs_persistent_anomaly(self):
        """Verify transition from TEMPORARY to PERSISTENT anomaly after persistence_count."""
        # Establish solid baseline at 65°C across 10 samples (window capacity)
        for i in range(10):
            self.detector.detect(self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=i)))

        # 1st anomalous reading -> TEMPORARY
        r1 = self.detector.detect(self._create_processed_data(temp=90.0, timestamp=self.now + timedelta(seconds=10)))
        self.assertEqual(r1.persistence, AnomalyPersistence.TEMPORARY)
        self.assertEqual(r1.status, AnomalyStatus.POSSIBLE_ANOMALY)

        # 2nd anomalous reading -> TEMPORARY
        r2 = self.detector.detect(self._create_processed_data(temp=91.0, timestamp=self.now + timedelta(seconds=11)))
        self.assertEqual(r2.persistence, AnomalyPersistence.TEMPORARY)

        # 3rd anomalous reading (meets persistence_count=3) -> PERSISTENT
        r3 = self.detector.detect(self._create_processed_data(temp=92.0, timestamp=self.now + timedelta(seconds=12)))
        self.assertEqual(r3.persistence, AnomalyPersistence.PERSISTENT)
        self.assertEqual(r3.status, AnomalyStatus.ANOMALY)


    # -------------------------------------------------------------------------
    # 6. Recovery & Hysteresis Tests (41-45)
    # -------------------------------------------------------------------------
    def test_anomaly_recovery_flow(self):
        """Test full anomaly recovery lifecycle: NORMAL -> ANOMALY -> RECOVERED -> NORMAL."""
        # Baseline
        for i in range(5):
            self.detector.detect(self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=i)))

        # Cause persistent anomaly
        for i in range(5, 8):
            self.detector.detect(self._create_processed_data(temp=85.0, timestamp=self.now + timedelta(seconds=i)))

        # Drop back to normal temp (65.0°C) -> ANOMALY_RECOVERED
        rec = self.detector.detect(self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=8)))
        self.assertEqual(rec.status, AnomalyStatus.RECOVERED)
        self.assertEqual(rec.transition, AnomalyTransition.ANOMALY_RECOVERED)

        # Subsequent normal temp -> NORMAL (no repeated recovery)
        norm = self.detector.detect(self._create_processed_data(temp=65.0, timestamp=self.now + timedelta(seconds=9)))
        self.assertEqual(norm.status, AnomalyStatus.NORMAL)
        self.assertEqual(norm.transition, AnomalyTransition.NO_CHANGE)

    # -------------------------------------------------------------------------
    # 7. Multi-Sensor Isolation Tests (51-53)
    # -------------------------------------------------------------------------
    def test_multi_sensor_isolation(self):
        """Verify sensor A and sensor B have completely independent baselines and state tracking."""
        # Sensor A at 65°C baseline
        for i in range(5):
            self.detector.detect(self._create_processed_data(sensor_id="sensor_A", temp=65.0, timestamp=self.now + timedelta(seconds=i)))

        # Sensor B at 82°C baseline
        for i in range(5):
            self.detector.detect(self._create_processed_data(sensor_id="sensor_B", temp=82.0, timestamp=self.now + timedelta(seconds=i)))

        res_a = self.detector.get_baseline("sensor_A")
        res_b = self.detector.get_baseline("sensor_B")

        self.assertEqual(res_a, 65.0)
        self.assertEqual(res_b, 82.0)

    # -------------------------------------------------------------------------
    # 8. Edge Case & Validation Tests (54-60)
    # -------------------------------------------------------------------------
    def test_invalid_input_rejections(self):
        """Verify detector rejects non-ProcessedThermalData or NaN/Infinite values."""
        with self.assertRaises(InvalidReadingError):
            self.detector.detect("invalid_input_string")  # type: ignore

        with self.assertRaises(InvalidReadingError):
            self.detector.detect(self._create_processed_data(temp=float("nan")))

        with self.assertRaises(InvalidReadingError):
            self.detector.detect(self._create_processed_data(temp=float("inf")))

    def test_config_validation_rejections(self):
        """Verify ThermalConfig rejects invalid Task 4 parameters."""
        with self.assertRaises(ValueError):
            ThermalConfig(minimum_baseline_samples=25, baseline_window_size=10)

        with self.assertRaises(ValueError):
            ThermalConfig(anomaly_sensitivity="SUPER_HIGH")

        with self.assertRaises(ValueError):
            ThermalConfig(anomaly_score_threshold=150.0)

    # -------------------------------------------------------------------------
    # 9. Pipeline Integration & Multi-Task Tests (61-67)
    # -------------------------------------------------------------------------
    def test_pipeline_task3_and_task4_coexistence(self):
        """Test parallel evaluation of Task 3 (Fixed Threshold) and Task 4 (Statistical Anomaly)."""
        pipe_config = ThermalConfig(
            warning_temperature=70.0,
            critical_temperature=90.0,
            overheating_temperature=90.0,
            baseline_window_size=10,
            minimum_baseline_samples=5,
            anomaly_score_threshold=50.0,
            smoothing_enabled=False,
        )
        processor = ThermalDataProcessor(pipe_config)
        threshold_detector = ThermalDetector(pipe_config)
        anomaly_detector = ThermalAnomalyDetector(pipe_config)

        # Baseline build at 45°C
        for i in range(5):
            r = ThermalReading(timestamp=self.now + timedelta(seconds=i), sensor_id="core_0", temperature=45.0)
            p = processor.process(r)
            threshold_detector.detect(p)
            anomaly_detector.detect(p)

        # Reading at 65°C: Below Task 3 warning threshold (70°C -> NORMAL state), but anomalous vs 45°C baseline
        r_test = ThermalReading(timestamp=self.now + timedelta(seconds=5), sensor_id="core_0", temperature=65.0)
        p_test = processor.process(r_test)

        task3_res = threshold_detector.detect(p_test)
        task4_res = anomaly_detector.detect(p_test)

        self.assertEqual(task3_res.state.value, "NORMAL")  # Below 70°C fixed safety limit
        self.assertIn(task4_res.status, (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY))  # Statistical anomaly vs 45°C baseline
        self.assertGreater(task4_res.anomaly_score, 50.0)

    def test_simulated_sensor_full_integration(self):
        """Test Task 1 -> Task 2 -> Task 4 with SimulatedThermalSensor stream."""
        acq = ThermalDataAcquisition(self.config)
        processor = ThermalDataProcessor(self.config)
        anomaly_detector = ThermalAnomalyDetector(self.config)

        sensor = SimulatedThermalSensor(sensor_id="core_0", start_temp=40.0, noise_std=0.1, seed=42)

        results = []
        for i in range(10):
            raw = acq.read_from_sensor(sensor)
            if raw:
                # Timestamps for clean rate calculation
                ts_raw = ThermalReading(temperature=raw.temperature, timestamp=self.now + timedelta(seconds=i), sensor_id=raw.sensor_id)
                p = processor.process(ts_raw)
                anom = anomaly_detector.detect(p)
                results.append(anom)

        self.assertEqual(len(results), 10)
        self.assertEqual(results[-1].status, AnomalyStatus.NORMAL)


if __name__ == "__main__":
    unittest.main()
