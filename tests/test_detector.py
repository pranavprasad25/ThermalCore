"""Unit test suite for ThermalDetector threshold & rule-based detection engine."""

from datetime import datetime, timezone, timedelta
import math
import unittest

from src.thermal.config import ThermalConfig
from src.thermal.detection_result import (
    StateTransition,
    ThermalCondition,
    ThermalDetectionResult,
    ThermalState,
)
from src.thermal.detector import ThermalDetector
from src.thermal.exceptions import InvalidReadingError
from src.thermal.processed_data import ProcessedThermalData, ThermalTrend
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading


class TestThermalDetector(unittest.TestCase):
    """Test suite covering ThermalDetector functionality, rules, transitions, and edge cases."""

    def setUp(self):
        """Set up standard configuration and detector instance for tests."""
        self.config = ThermalConfig(
            warning_temperature=70.0,
            critical_temperature=90.0,
            overheating_temperature=90.0,
            spike_threshold=10.0,
            rapid_rise_threshold=2.0,
            abnormal_cooling_threshold=5.0,
            sustained_high_temperature=80.0,
            sustained_high_duration=5.0,
        )
        self.detector = ThermalDetector(self.config)
        self.now = datetime.now(timezone.utc)

    def _create_processed_data(
        self,
        sensor_id: str = "core_0",
        temp: float = 45.0,
        raw_temp: float = 45.0,
        rate_of_change: float = 0.0,
        timestamp: datetime = None,
        trend: ThermalTrend = ThermalTrend.STABLE,
    ) -> ProcessedThermalData:
        """Helper factory to construct test ProcessedThermalData objects."""
        ts = timestamp if timestamp is not None else self.now
        reading = ThermalReading(
            timestamp=ts,
            sensor_id=sensor_id,
            temperature=raw_temp,
            unit="C",
        )
        return ProcessedThermalData(
            raw_reading=reading,
            processed_temperature=temp,
            moving_average=temp,
            rate_of_change=rate_of_change,
            trend=trend,
        )

    # -------------------------------------------------------------------------
    # 1. State Classification Tests
    # -------------------------------------------------------------------------
    def test_classify_normal_temperature(self):
        """Verify temperatures below warning threshold are classified as NORMAL."""
        self.assertEqual(self.detector.classify_temperature(25.0), ThermalState.NORMAL)
        self.assertEqual(self.detector.classify_temperature(69.9), ThermalState.NORMAL)

    def test_classify_warning_temperature(self):
        """Verify temperatures between warning and critical are classified as WARNING."""
        self.assertEqual(self.detector.classify_temperature(70.0), ThermalState.WARNING)
        self.assertEqual(self.detector.classify_temperature(85.0), ThermalState.WARNING)
        self.assertEqual(self.detector.classify_temperature(89.99), ThermalState.WARNING)

    def test_classify_critical_temperature(self):
        """Verify temperatures at or above critical threshold are classified as CRITICAL."""
        self.assertEqual(self.detector.classify_temperature(90.0), ThermalState.CRITICAL)
        self.assertEqual(self.detector.classify_temperature(105.0), ThermalState.CRITICAL)

    # -------------------------------------------------------------------------
    # 2. Overheating Detection Tests
    # -------------------------------------------------------------------------
    def test_evaluate_overheating_positive(self):
        """Verify evaluate_overheating returns True when temperature meets/exceeds threshold."""
        self.assertTrue(self.detector.evaluate_overheating(90.0))
        self.assertTrue(self.detector.evaluate_overheating(95.5))

    def test_evaluate_overheating_negative(self):
        """Verify evaluate_overheating returns False when temperature is below threshold."""
        self.assertFalse(self.detector.evaluate_overheating(89.9))
        self.assertFalse(self.detector.evaluate_overheating(50.0))

    def test_detect_overheating_condition(self):
        """Verify detect includes OVERHEATING condition and explanatory reason."""
        data = self._create_processed_data(temp=92.0, raw_temp=92.0)
        res = self.detector.detect(data)
        self.assertIn(ThermalCondition.OVERHEATING, res.conditions)
        self.assertEqual(res.state, ThermalState.CRITICAL)
        self.assertTrue(any("overheating threshold" in r for r in res.reasons))

    # -------------------------------------------------------------------------
    # 3. Sudden Spike Detection Tests
    # -------------------------------------------------------------------------
    def test_evaluate_spike_positive(self):
        """Verify evaluate_spike returns True when rate of change >= spike_threshold."""
        self.assertTrue(self.detector.evaluate_spike(10.0))
        self.assertTrue(self.detector.evaluate_spike(15.5))

    def test_evaluate_spike_negative(self):
        """Verify evaluate_spike returns False when rate of change < spike_threshold."""
        self.assertFalse(self.detector.evaluate_spike(9.99))
        self.assertFalse(self.detector.evaluate_spike(0.0))

    def test_detect_spike_condition(self):
        """Verify detect includes SUDDEN_SPIKE condition."""
        data = self._create_processed_data(temp=50.0, rate_of_change=12.0)
        res = self.detector.detect(data)
        self.assertIn(ThermalCondition.SUDDEN_SPIKE, res.conditions)
        self.assertTrue(any("spike threshold" in r for r in res.reasons))

    # -------------------------------------------------------------------------
    # 4. Rapid Rise Detection Tests
    # -------------------------------------------------------------------------
    def test_evaluate_rapid_rise_positive(self):
        """Verify evaluate_rapid_rise returns True for rates between rapid_rise and spike thresholds."""
        self.assertTrue(self.detector.evaluate_rapid_rise(2.0))
        self.assertTrue(self.detector.evaluate_rapid_rise(5.0))
        self.assertTrue(self.detector.evaluate_rapid_rise(9.9))

    def test_evaluate_rapid_rise_negative(self):
        """Verify evaluate_rapid_rise returns False outside boundary range."""
        self.assertFalse(self.detector.evaluate_rapid_rise(1.99))
        self.assertFalse(self.detector.evaluate_rapid_rise(10.0))  # 10.0 is spike

    def test_detect_rapid_rise_condition(self):
        """Verify detect includes RAPID_RISE condition."""
        data = self._create_processed_data(temp=50.0, rate_of_change=4.5)
        res = self.detector.detect(data)
        self.assertIn(ThermalCondition.RAPID_RISE, res.conditions)
        self.assertNotIn(ThermalCondition.SUDDEN_SPIKE, res.conditions)

    # -------------------------------------------------------------------------
    # 5. Abnormal Cooling Detection Tests
    # -------------------------------------------------------------------------
    def test_evaluate_abnormal_cooling_positive(self):
        """Verify evaluate_abnormal_cooling returns True for steep negative rates."""
        self.assertTrue(self.detector.evaluate_abnormal_cooling(-5.0))
        self.assertTrue(self.detector.evaluate_abnormal_cooling(-8.2))

    def test_evaluate_abnormal_cooling_negative(self):
        """Verify evaluate_abnormal_cooling returns False for mild negative or positive rates."""
        self.assertFalse(self.detector.evaluate_abnormal_cooling(-4.9))
        self.assertFalse(self.detector.evaluate_abnormal_cooling(0.0))

    def test_detect_abnormal_cooling_condition(self):
        """Verify detect includes ABNORMAL_COOLING condition."""
        data = self._create_processed_data(temp=40.0, rate_of_change=-6.0)
        res = self.detector.detect(data)
        self.assertIn(ThermalCondition.ABNORMAL_COOLING, res.conditions)

    # -------------------------------------------------------------------------
    # 6. Sustained High Temperature Tests
    # -------------------------------------------------------------------------
    def test_sustained_high_duration_trigger(self):
        """Verify SUSTAINED_HIGH triggers after sustained duration and resets when temp drops."""
        sensor_id = "core_0"
        t0 = self.now

        # Initial reading at 82.0°C (t = 0s) -> Start tracking
        d0 = self._create_processed_data(sensor_id=sensor_id, temp=82.0, timestamp=t0)
        res0 = self.detector.detect(d0)
        self.assertNotIn(ThermalCondition.SUSTAINED_HIGH, res0.conditions)

        # Reading after 3s (t = 3s < 5s) -> Not yet sustained
        t3 = t0 + timedelta(seconds=3)
        d3 = self._create_processed_data(sensor_id=sensor_id, temp=83.0, timestamp=t3)
        res3 = self.detector.detect(d3)
        self.assertNotIn(ThermalCondition.SUSTAINED_HIGH, res3.conditions)

        # Reading after 5s (t = 5s >= 5s) -> Trigger SUSTAINED_HIGH
        t5 = t0 + timedelta(seconds=5)
        d5 = self._create_processed_data(sensor_id=sensor_id, temp=81.0, timestamp=t5)
        res5 = self.detector.detect(d5)
        self.assertIn(ThermalCondition.SUSTAINED_HIGH, res5.conditions)

        # Temperature drops below threshold (75°C) -> Reset sustained tracking
        t6 = t0 + timedelta(seconds=6)
        d6 = self._create_processed_data(sensor_id=sensor_id, temp=75.0, timestamp=t6)
        res6 = self.detector.detect(d6)
        self.assertNotIn(ThermalCondition.SUSTAINED_HIGH, res6.conditions)

        # Temperature goes back up (82°C) at t = 7s -> Fresh tracking starts
        t7 = t0 + timedelta(seconds=7)
        d7 = self._create_processed_data(sensor_id=sensor_id, temp=82.0, timestamp=t7)
        res7 = self.detector.detect(d7)
        self.assertNotIn(ThermalCondition.SUSTAINED_HIGH, res7.conditions)

    # -------------------------------------------------------------------------
    # 7. Recovery Detection & State Transition Tests
    # -------------------------------------------------------------------------
    def test_state_transitions_flow(self):
        """Test full transition sequence: NORMAL -> WARNING -> CRITICAL -> WARNING -> NORMAL."""
        sensor = "core_0"

        # 1. Initial NORMAL
        d1 = self._create_processed_data(sensor_id=sensor, temp=40.0)
        r1 = self.detector.detect(d1)
        self.assertEqual(r1.state, ThermalState.NORMAL)
        self.assertIsNone(r1.previous_state)
        self.assertEqual(r1.transition, StateTransition.NO_CHANGE)

        # 2. Escalation to WARNING
        d2 = self._create_processed_data(sensor_id=sensor, temp=75.0)
        r2 = self.detector.detect(d2)
        self.assertEqual(r2.state, ThermalState.WARNING)
        self.assertEqual(r2.previous_state, ThermalState.NORMAL)
        self.assertEqual(r2.transition, StateTransition.ESCALATED)

        # 3. Escalation to CRITICAL
        d3 = self._create_processed_data(sensor_id=sensor, temp=92.0)
        r3 = self.detector.detect(d3)
        self.assertEqual(r3.state, ThermalState.CRITICAL)
        self.assertEqual(r3.previous_state, ThermalState.WARNING)
        self.assertEqual(r3.transition, StateTransition.ESCALATED)

        # 4. De-escalation to WARNING
        d4 = self._create_processed_data(sensor_id=sensor, temp=78.0)
        r4 = self.detector.detect(d4)
        self.assertEqual(r4.state, ThermalState.WARNING)
        self.assertEqual(r4.previous_state, ThermalState.CRITICAL)
        self.assertEqual(r4.transition, StateTransition.DE_ESCALATED)

        # 5. Recovery to NORMAL
        d5 = self._create_processed_data(sensor_id=sensor, temp=45.0)
        r5 = self.detector.detect(d5)
        self.assertEqual(r5.state, ThermalState.NORMAL)
        self.assertEqual(r5.previous_state, ThermalState.WARNING)
        self.assertEqual(r5.transition, StateTransition.RECOVERED)
        self.assertIn(ThermalCondition.RECOVERY, r5.conditions)

    # -------------------------------------------------------------------------
    # 8. Multiple Conditions Combination Tests
    # -------------------------------------------------------------------------
    def test_multiple_simultaneous_conditions(self):
        """Verify detector handles simultaneous Overheating and Spike conditions."""
        data = self._create_processed_data(temp=95.0, rate_of_change=15.0)
        res = self.detector.detect(data)
        self.assertEqual(res.state, ThermalState.CRITICAL)
        self.assertIn(ThermalCondition.OVERHEATING, res.conditions)
        self.assertIn(ThermalCondition.SUDDEN_SPIKE, res.conditions)
        self.assertGreaterEqual(len(res.reasons), 2)

    # -------------------------------------------------------------------------
    # 9. Configuration & Parameter Validation Tests
    # -------------------------------------------------------------------------
    def test_config_invalid_thresholds(self):
        """Verify ThermalConfig rejects invalid threshold relationships."""
        with self.assertRaises(ValueError):
            ThermalConfig(warning_temperature=90.0, critical_temperature=70.0)

        with self.assertRaises(ValueError):
            ThermalConfig(spike_threshold=-5.0)

        with self.assertRaises(ValueError):
            ThermalConfig(sustained_high_duration=0.0)

    def test_detector_invalid_input_type(self):
        """Verify detect raises InvalidReadingError for invalid inputs."""
        with self.assertRaises(InvalidReadingError):
            self.detector.detect("invalid_data_object")  # type: ignore

    def test_detector_nan_temperature(self):
        """Verify detect raises InvalidReadingError for NaN temperature values."""
        data = self._create_processed_data(temp=float("nan"))
        with self.assertRaises(InvalidReadingError):
            self.detector.detect(data)

    def test_detector_infinite_temperature(self):
        """Verify detect raises InvalidReadingError for Infinite temperature values."""
        data = self._create_processed_data(temp=float("inf"))
        with self.assertRaises(InvalidReadingError):
            self.detector.detect(data)

    # -------------------------------------------------------------------------
    # 10. Multi-Sensor Isolation Tests
    # -------------------------------------------------------------------------
    def test_multi_sensor_isolation(self):
        """Verify state tracking for core_0 and core_1 are completely independent."""
        # core_0 goes to CRITICAL
        d_core0 = self._create_processed_data(sensor_id="core_0", temp=95.0)
        r_core0 = self.detector.detect(d_core0)
        self.assertEqual(r_core0.state, ThermalState.CRITICAL)

        # core_1 evaluated at NORMAL -> should not inherit core_0's previous state
        d_core1 = self._create_processed_data(sensor_id="core_1", temp=45.0)
        r_core1 = self.detector.detect(d_core1)
        self.assertEqual(r_core1.state, ThermalState.NORMAL)
        self.assertIsNone(r_core1.previous_state)
        self.assertEqual(r_core1.transition, StateTransition.NO_CHANGE)

        # Reset core_0 only
        self.detector.reset(sensor_id="core_0")
        d_core0_new = self._create_processed_data(sensor_id="core_0", temp=45.0)
        r_core0_new = self.detector.detect(d_core0_new)
        self.assertIsNone(r_core0_new.previous_state)

    # -------------------------------------------------------------------------
    # 11. Serialization & Dictionary Representation
    # -------------------------------------------------------------------------
    def test_detection_result_to_dict(self):
        """Verify ThermalDetectionResult.to_dict produces clean serializable dictionary."""
        data = self._create_processed_data(temp=95.0, rate_of_change=12.0)
        res = self.detector.detect(data)
        res_dict = res.to_dict()

        self.assertEqual(res_dict["sensor_id"], "core_0")
        self.assertEqual(res_dict["temperature"], 95.0)
        self.assertEqual(res_dict["state"], "CRITICAL")
        self.assertIn("OVERHEATING", res_dict["conditions"])
        self.assertIn("SUDDEN_SPIKE", res_dict["conditions"])
        self.assertIsInstance(res_dict["reasons"], list)

    # -------------------------------------------------------------------------
    # 12. End-to-End Pipeline Integration Test
    # -------------------------------------------------------------------------
    def test_pipeline_integration(self):
        """Test full end-to-end flow: Raw Reading -> Processor -> Detector."""
        pipe_config = ThermalConfig(
            warning_temperature=70.0,
            critical_temperature=90.0,
            overheating_temperature=90.0,
            spike_threshold=10.0,
            smoothing_enabled=False,
        )
        processor = ThermalDataProcessor(pipe_config)
        detector = ThermalDetector(pipe_config)

        # Reading 1: 40°C
        r1 = ThermalReading(timestamp=self.now, sensor_id="core_0", temperature=40.0)
        p1 = processor.process(r1)
        det1 = detector.detect(p1)
        self.assertEqual(det1.state, ThermalState.NORMAL)

        # Reading 2: 95°C (1 second later) -> Spike + Overheating
        r2 = ThermalReading(timestamp=self.now + timedelta(seconds=1), sensor_id="core_0", temperature=95.0)
        p2 = processor.process(r2)
        det2 = detector.detect(p2)
        self.assertEqual(det2.state, ThermalState.CRITICAL)
        self.assertIn(ThermalCondition.OVERHEATING, det2.conditions)
        self.assertIn(ThermalCondition.SUDDEN_SPIKE, det2.conditions)
        self.assertEqual(det2.transition, StateTransition.ESCALATED)



if __name__ == "__main__":
    unittest.main()
