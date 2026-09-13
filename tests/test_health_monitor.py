"""Automated unit test suite for ThermalHealthMonitor engine."""

from datetime import datetime, timedelta, timezone
import math
from typing import Optional
import unittest

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.anomaly_result import AnomalyPersistence, AnomalySeverity, AnomalyStatus
from src.thermal.config import ThermalConfig
from src.thermal.detection_result import StateTransition, ThermalCondition, ThermalState
from src.thermal.exceptions import InvalidReadingError
from src.thermal.health_monitor import ThermalHealthMonitor
from src.thermal.health_result import ThermalHealthResult, ThermalHealthStatus
from src.thermal.processed_data import ThermalTrend
from src.thermal.reading import ThermalReading
from src.thermal.sensor import SimulatedThermalSensor


class TestThermalHealthMonitor(unittest.TestCase):
    """Test suite covering ThermalHealthMonitor orchestration, scoring, status, alerts, and accessors."""

    def setUp(self):
        """Set up standard configuration and health monitor instance."""
        self.config = ThermalConfig(
            warning_temperature=70.0,
            critical_temperature=90.0,
            baseline_window_size=10,
            minimum_baseline_samples=5,
            anomaly_score_threshold=50.0,
            smoothing_enabled=False,
        )
        self.monitor = ThermalHealthMonitor(self.config)
        self.now = datetime.now(timezone.utc)

    def _create_reading(
        self,
        sensor_id: str = "core_0",
        temp: float = 45.0,
        timestamp: Optional[datetime] = None,
    ) -> ThermalReading:
        """Helper factory to create ThermalReading instances."""
        ts = timestamp if timestamp is not None else self.now
        return ThermalReading(
            timestamp=ts,
            sensor_id=sensor_id,
            temperature=temp,
            unit="C",
        )

    # -------------------------------------------------------------------------
    # 1. Accessors Before Data & Initial State
    # -------------------------------------------------------------------------
    def test_getters_initial_empty_state(self):
        """Verify accessors return appropriate empty/UNKNOWN defaults prior to processing readings."""
        sensor_id = "core_0"
        self.assertEqual(self.monitor.get_health_status(sensor_id), ThermalHealthStatus.UNKNOWN)
        self.assertIsNone(self.monitor.get_health_score(sensor_id))
        self.assertIsNone(self.monitor.get_current_temperature(sensor_id))
        self.assertEqual(self.monitor.get_active_alerts(sensor_id), [])
        self.assertIsNone(self.monitor.get_anomalies(sensor_id))
        self.assertIsNone(self.monitor.get_latest_result(sensor_id))
        self.assertEqual(self.monitor.get_history(sensor_id), [])

    # -------------------------------------------------------------------------
    # 2. Update Flow & Health Status Classification
    # -------------------------------------------------------------------------
    def test_update_healthy_baseline(self):
        """Verify update returns HEALTHY result for normal baseline readings."""
        r = self._create_reading(temp=45.0)
        res = self.monitor.update(r)

        self.assertIsInstance(res, ThermalHealthResult)
        self.assertEqual(res.sensor_id, "core_0")
        self.assertEqual(res.temperature, 45.0)
        self.assertEqual(res.health_status, ThermalHealthStatus.HEALTHY)
        self.assertEqual(res.health_score, 100.0)
        self.assertEqual(res.active_alerts, [])

    def test_update_warning_operating_state(self):
        """Verify update transitions to WARNING status when Task 3 warning threshold (70°C) is reached."""
        # Baseline at 68°C
        for i in range(5):
            self.monitor.update(self._create_reading(temp=68.0, timestamp=self.now + timedelta(seconds=i)))

        # Jump to 75°C (WARNING threshold)
        r_warn = self._create_reading(temp=75.0, timestamp=self.now + timedelta(seconds=5))
        res_warn = self.monitor.update(r_warn)

        self.assertEqual(res_warn.thermal_state, ThermalState.WARNING)
        self.assertIn(res_warn.health_status, (ThermalHealthStatus.WARNING, ThermalHealthStatus.CRITICAL))
        self.assertLess(res_warn.health_score, 100.0)

    def test_update_critical_operating_state(self):
        """Verify update transitions to CRITICAL status when Task 3 critical threshold (90°C) is reached."""
        r_crit = self._create_reading(temp=95.0)
        res_crit = self.monitor.update(r_crit)

        self.assertEqual(res_crit.thermal_state, ThermalState.CRITICAL)
        self.assertEqual(res_crit.health_status, ThermalHealthStatus.CRITICAL)
        self.assertLessEqual(res_crit.health_score, 50.0)

    # -------------------------------------------------------------------------
    # 3. Active Alerts Aggregation & Clearing
    # -------------------------------------------------------------------------
    def test_active_alerts_aggregation(self):
        """Verify active alerts aggregate unique Task 3 conditions and Task 4 anomaly alerts."""
        # Baseline
        for i in range(5):
            self.monitor.update(self._create_reading(temp=45.0, timestamp=self.now + timedelta(seconds=i)))

        # Sudden jump to 95°C -> Overheating, Spike, Anomaly
        r_jump = self._create_reading(temp=95.0, timestamp=self.now + timedelta(seconds=5))
        res = self.monitor.update(r_jump)

        self.assertIn("OVERHEATING", res.active_alerts)
        self.assertIn("SUDDEN_SPIKE", res.active_alerts)
        self.assertIn("THERMAL_ANOMALY", res.active_alerts)
        self.assertEqual(len(res.active_alerts), len(set(res.active_alerts)))  # De-duplicated

    def test_active_alerts_clearing_on_recovery(self):
        """Verify active alerts clear when temperature returns to normal and settles."""
        # Cause active alert
        self.monitor.update(self._create_reading(temp=95.0))
        self.assertGreater(len(self.monitor.get_active_alerts("core_0")), 0)

        # Drop back to 45°C
        self.monitor.update(self._create_reading(temp=45.0, timestamp=self.now + timedelta(seconds=1)))

        # Settle at 45°C (no abnormal cooling rate on 2nd step)
        self.monitor.update(self._create_reading(temp=45.0, timestamp=self.now + timedelta(seconds=2)))
        self.assertEqual(self.monitor.get_active_alerts("core_0"), [])


    # -------------------------------------------------------------------------
    # 4. Health Score Penalties & Bounds
    # -------------------------------------------------------------------------
    def test_health_score_penalties_breakdown(self):
        """Verify health score applies explicit penalties and populates penalty breakdown array."""
        r = self._create_reading(temp=75.0)
        res = self.monitor.update(r)

        self.assertLess(res.health_score, 100.0)
        self.assertGreater(len(res.penalties), 0)
        self.assertTrue(any("WARNING operating temperature state" in p for p in res.penalties))

    def test_health_score_bounded_range(self):
        """Verify health score is strictly bounded between 0.0 and 100.0."""
        # Extreme stress reading
        r = self._create_reading(temp=150.0)
        res = self.monitor.update(r)

        self.assertGreaterEqual(res.health_score, 0.0)
        self.assertLessEqual(res.health_score, 100.0)

    # -------------------------------------------------------------------------
    # 5. Read and Update Integration (Task 1 + Task 5)
    # -------------------------------------------------------------------------
    def test_read_and_update_with_simulated_sensor(self):
        """Verify read_and_update acquires reading from SimulatedThermalSensor and executes update."""
        sensor = SimulatedThermalSensor(sensor_id="core_0", start_temp=45.0, seed=42)
        res = self.monitor.read_and_update(sensor)

        self.assertIsNotNone(res)
        self.assertEqual(res.sensor_id, "core_0")
        self.assertGreater(res.temperature, 0.0)

    # -------------------------------------------------------------------------
    # 6. Multi-Sensor State Isolation
    # -------------------------------------------------------------------------
    def test_multi_sensor_isolation(self):
        """Verify Core 0 and Core 1 maintain independent health state and history."""
        # Core 0 at CRITICAL (95°C)
        res_c0 = self.monitor.update(self._create_reading(sensor_id="core_0", temp=95.0))
        self.assertEqual(res_c0.health_status, ThermalHealthStatus.CRITICAL)

        # Core 1 at HEALTHY (45°C)
        res_c1 = self.monitor.update(self._create_reading(sensor_id="core_1", temp=45.0))
        self.assertEqual(res_c1.health_status, ThermalHealthStatus.HEALTHY)

        self.assertEqual(self.monitor.get_health_status("core_0"), ThermalHealthStatus.CRITICAL)
        self.assertEqual(self.monitor.get_health_status("core_1"), ThermalHealthStatus.HEALTHY)

    # -------------------------------------------------------------------------
    # 7. Reset Behavior
    # -------------------------------------------------------------------------
    def test_reset_clears_state_without_destroying_config(self):
        """Verify reset clears runtime health state while preserving configuration."""
        self.monitor.update(self._create_reading(sensor_id="core_0", temp=95.0))
        self.assertEqual(self.monitor.get_health_status("core_0"), ThermalHealthStatus.CRITICAL)

        self.monitor.reset("core_0")
        self.assertEqual(self.monitor.get_health_status("core_0"), ThermalHealthStatus.UNKNOWN)
        self.assertEqual(self.monitor.config.warning_temperature, 70.0)  # Config intact

    # -------------------------------------------------------------------------
    # 8. Invalid Reading Error Rejection
    # -------------------------------------------------------------------------
    def test_update_invalid_reading_rejection(self):
        """Verify update raises InvalidReadingError when passed non-ThermalReading object."""
        with self.assertRaises(InvalidReadingError):
            self.monitor.update("not_a_reading_object")  # type: ignore

    # -------------------------------------------------------------------------
    # 9. Serialization Output
    # -------------------------------------------------------------------------
    def test_health_result_to_dict(self):
        """Verify ThermalHealthResult.to_dict produces clean serializable dictionary."""
        r = self._create_reading(temp=75.0)
        res = self.monitor.update(r)
        res_dict = res.to_dict()

        self.assertEqual(res_dict["sensor_id"], "core_0")
        self.assertEqual(res_dict["temperature"], 75.0)
        self.assertIn("active_alerts", res_dict)
        self.assertIn("penalties", res_dict)
        self.assertIsInstance(res_dict["penalties"], list)


if __name__ == "__main__":
    unittest.main()
