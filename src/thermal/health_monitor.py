"""Central Thermal Health Monitoring Engine for ThermoShift / ThermalCore."""

from typing import Dict, List, Optional

from src.thermal.acquisition import ThermalDataAcquisition
from src.thermal.anomaly_detector import ThermalAnomalyDetector
from src.thermal.anomaly_result import AnomalyPersistence, AnomalySeverity, AnomalyStatus, ThermalAnomalyResult
from src.thermal.config import ThermalConfig
from src.thermal.detection_result import ThermalCondition, ThermalState
from src.thermal.detector import ThermalDetector
from src.thermal.exceptions import InvalidReadingError
from src.thermal.health_result import ThermalHealthResult, ThermalHealthStatus
from src.thermal.processor import ThermalDataProcessor
from src.thermal.reading import ThermalReading
from src.thermal.sensor import ThermalSensor


class ThermalHealthMonitor:
    """Central orchestrator coordinating acquisition, processing, rule detection, and anomaly detection.

    Provides a clean unified API for assessing system thermal health, scores, active alerts,
    and history without callers needing to manually coordinate individual pipeline stages.
    """

    def __init__(
        self,
        config: Optional[ThermalConfig] = None,
        processor: Optional[ThermalDataProcessor] = None,
        detector: Optional[ThermalDetector] = None,
        anomaly_detector: Optional[ThermalAnomalyDetector] = None,
    ):
        """Initialize ThermalHealthMonitor with configuration and pipeline stage components."""
        self.config = config if config is not None else ThermalConfig()
        self.processor = processor if processor is not None else ThermalDataProcessor(self.config)
        self.detector = detector if detector is not None else ThermalDetector(self.config)
        self.anomaly_detector = anomaly_detector if anomaly_detector is not None else ThermalAnomalyDetector(self.config)
        self._acquisition = ThermalDataAcquisition(self.config)
        self._latest_results: Dict[str, ThermalHealthResult] = {}
        self._latest_anomalies: Dict[str, ThermalAnomalyResult] = {}

    def update(self, reading: ThermalReading) -> ThermalHealthResult:
        """Process a single ThermalReading through the complete thermal health monitoring pipeline.

        Args:
            reading: Incoming ThermalReading instance.

        Returns:
            ThermalHealthResult: Unified health evaluation result.

        Raises:
            InvalidReadingError: If reading is invalid or malformed.
        """
        if not isinstance(reading, ThermalReading):
            raise InvalidReadingError(f"Health monitor accepts ThermalReading instances, got {type(reading).__name__}")

        sensor_id = reading.sensor_id

        # 1. Processing & History (Task 2)
        processed_data = self.processor.process(reading)

        # 2. Fixed Threshold & Rule Detection (Task 3)
        threshold_result = self.detector.detect(processed_data)

        # 3. Statistical Anomaly Detection (Task 4)
        anomaly_result = self.anomaly_detector.detect(processed_data)
        self._latest_anomalies[sensor_id] = anomaly_result

        # 4. Aggregate Active Alerts
        active_alerts: List[str] = []
        for condition in threshold_result.conditions:
            if condition.value not in active_alerts:
                active_alerts.append(condition.value)

        if anomaly_result.status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY):
            if "THERMAL_ANOMALY" not in active_alerts:
                active_alerts.append("THERMAL_ANOMALY")

        if anomaly_result.persistence == AnomalyPersistence.PERSISTENT:
            if "PERSISTENT_ANOMALY" not in active_alerts:
                active_alerts.append("PERSISTENT_ANOMALY")

        # 5. Compute Health Score & Penalties Breakdown
        base_score = 100.0
        penalties: List[str] = []

        # Operating state penalties
        if threshold_result.state == ThermalState.CRITICAL:
            base_score -= 50.0
            penalties.append("CRITICAL operating temperature state (-50.0 pts)")
        elif threshold_result.state == ThermalState.WARNING:
            base_score -= 20.0
            penalties.append("WARNING operating temperature state (-20.0 pts)")

        # Anomaly severity penalties
        if anomaly_result.severity == AnomalySeverity.CRITICAL:
            base_score -= 40.0
            penalties.append("CRITICAL statistical anomaly severity (-40.0 pts)")
        elif anomaly_result.severity == AnomalySeverity.HIGH:
            base_score -= 25.0
            penalties.append("HIGH statistical anomaly severity (-25.0 pts)")
        elif anomaly_result.severity == AnomalySeverity.MEDIUM:
            base_score -= 15.0
            penalties.append("MEDIUM statistical anomaly severity (-15.0 pts)")
        elif anomaly_result.severity == AnomalySeverity.LOW:
            base_score -= 5.0
            penalties.append("LOW statistical anomaly severity (-5.0 pts)")

        # Persistence penalty
        if anomaly_result.persistence == AnomalyPersistence.PERSISTENT:
            base_score -= 10.0
            penalties.append("Persistent anomalous thermal behavior (-10.0 pts)")

        # Condition specific penalties
        if ThermalCondition.OVERHEATING in threshold_result.conditions:
            base_score -= 10.0
            penalties.append("Overheating condition active (-10.0 pts)")
        if ThermalCondition.SUDDEN_SPIKE in threshold_result.conditions:
            base_score -= 10.0
            penalties.append("Sudden temperature spike active (-10.0 pts)")
        if ThermalCondition.RAPID_RISE in threshold_result.conditions:
            base_score -= 5.0
            penalties.append("Rapid temperature rise active (-5.0 pts)")
        if ThermalCondition.SUSTAINED_HIGH in threshold_result.conditions:
            base_score -= 10.0
            penalties.append("Sustained high temperature condition (-10.0 pts)")

        health_score = float(max(0.0, min(100.0, base_score)))

        # 6. Derive Overall Thermal Health Status
        if threshold_result.state == ThermalState.CRITICAL or anomaly_result.severity == AnomalySeverity.CRITICAL:
            health_status = ThermalHealthStatus.CRITICAL
        elif threshold_result.state == ThermalState.WARNING or anomaly_result.severity == AnomalySeverity.HIGH or health_score < 60.0:
            health_status = ThermalHealthStatus.WARNING
        elif anomaly_result.severity == AnomalySeverity.MEDIUM or len(active_alerts) > 0 or health_score < 85.0:
            health_status = ThermalHealthStatus.DEGRADED
        else:
            health_status = ThermalHealthStatus.HEALTHY

        # 7. Merge Unified Explainability Reasons
        reasons: List[str] = list(threshold_result.reasons) + list(anomaly_result.reasons)

        # Construct Unified Result
        result = ThermalHealthResult(
            timestamp=reading.timestamp,
            sensor_id=sensor_id,
            temperature=processed_data.processed_temperature,
            raw_temperature=reading.temperature,
            trend=processed_data.trend,
            rate_of_change=processed_data.rate_of_change,
            thermal_state=threshold_result.state,
            anomaly_status=anomaly_result.status,
            anomaly_severity=anomaly_result.severity,
            anomaly_score=anomaly_result.anomaly_score,
            health_score=health_score,
            health_status=health_status,
            active_alerts=active_alerts,
            state_transition=threshold_result.transition,
            anomaly_transition=anomaly_result.transition,
            reasons=reasons,
            penalties=penalties,
        )

        self._latest_results[sensor_id] = result
        return result

    def read_and_update(self, sensor: ThermalSensor) -> Optional[ThermalHealthResult]:
        """Acquire a reading from sensor (Task 1) and execute the update pipeline.

        Args:
            sensor: ThermalSensor instance to acquire reading from.

        Returns:
            Optional[ThermalHealthResult]: Updated result or None if read fails.
        """
        reading = self._acquisition.read_from_sensor(sensor)
        if reading is not None:
            return self.update(reading)
        return None

    def get_latest_result(self, sensor_id: str) -> Optional[ThermalHealthResult]:
        """Retrieve the most recent ThermalHealthResult for sensor_id."""
        return self._latest_results.get(sensor_id)

    def get_health_status(self, sensor_id: str) -> ThermalHealthStatus:
        """Retrieve overall health status for sensor_id, returning UNKNOWN if no data exists."""
        res = self._latest_results.get(sensor_id)
        return res.health_status if res is not None else ThermalHealthStatus.UNKNOWN

    def get_health_score(self, sensor_id: str) -> Optional[float]:
        """Retrieve latest health score for sensor_id, returning None if no data exists."""
        res = self._latest_results.get(sensor_id)
        return res.health_score if res is not None else None

    def get_current_temperature(self, sensor_id: str) -> Optional[float]:
        """Retrieve current processed temperature for sensor_id from processor."""
        try:
            return self.processor.get_current_temperature(sensor_id)
        except InvalidReadingError:
            return None

    def get_active_alerts(self, sensor_id: str) -> List[str]:
        """Retrieve currently active alerts for sensor_id."""
        res = self._latest_results.get(sensor_id)
        return list(res.active_alerts) if res is not None else []

    def get_anomalies(self, sensor_id: str) -> Optional[ThermalAnomalyResult]:
        """Retrieve the latest Task 4 Anomaly Result for sensor_id."""
        return self._latest_anomalies.get(sensor_id)

    def get_history(self, sensor_id: str) -> List[ThermalReading]:
        """Retrieve full raw reading history for sensor_id from Task 2 processor."""
        return self.processor.get_history(sensor_id)

    def reset(self, sensor_id: Optional[str] = None) -> None:
        """Safely clear runtime state for a specific sensor or all sensors without destroying configuration.

        Args:
            sensor_id: Specific sensor ID to reset, or None to reset all sensors.
        """
        if sensor_id is None:
            self._latest_results.clear()
            self._latest_anomalies.clear()
        else:
            self._latest_results.pop(sensor_id, None)
            self._latest_anomalies.pop(sensor_id, None)

        self.processor.clear_history(sensor_id)
        self.detector.reset(sensor_id)
        self.anomaly_detector.reset(sensor_id)
