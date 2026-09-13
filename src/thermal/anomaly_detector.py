"""Deterministic statistical thermal anomaly detection engine."""

import math
import statistics
from typing import Dict, List, Optional

from src.thermal.config import ThermalConfig
from src.thermal.exceptions import InvalidReadingError
from src.thermal.history import ThermalHistoryBuffer
from src.thermal.processed_data import ProcessedThermalData
from src.thermal.anomaly_result import (
    AnomalyPersistence,
    AnomalySeverity,
    AnomalyStatus,
    AnomalyTransition,
    ThermalAnomalyResult,
)


class ThermalAnomalyDetector:
    """Deterministic statistical thermal anomaly detection engine.

    Identifies temperature behavior that is unusual compared with the system's
    rolling historical baseline, evaluating deviation, variability (z-score), rate anomaly,
    persistence, and recovery.
    """

    _SEVERITY_RANK = {
        AnomalySeverity.NONE: 0,
        AnomalySeverity.LOW: 1,
        AnomalySeverity.MEDIUM: 2,
        AnomalySeverity.HIGH: 3,
        AnomalySeverity.CRITICAL: 4,
    }

    _SENSITIVITY_MAP = {
        "LOW": 0.6,
        "MEDIUM": 1.0,
        "HIGH": 1.5,
    }

    def __init__(self, config: Optional[ThermalConfig] = None):
        """Initialize ThermalAnomalyDetector with configuration and history state."""
        self.config = config if config is not None else ThermalConfig()
        self.history = ThermalHistoryBuffer(max_capacity=self.config.baseline_window_size)
        self._previous_statuses: Dict[str, AnomalyStatus] = {}
        self._previous_severities: Dict[str, AnomalySeverity] = {}
        self._anomalous_counts: Dict[str, int] = {}
        self._anomalous_start_times: Dict[str, Optional[math.nan]] = {}

    def get_baseline(self, sensor_id: str) -> Optional[float]:
        """Compute robust rolling baseline temperature (median) for sensor_id."""
        readings = self.history.recent(sensor_id, self.config.baseline_window_size)
        if len(readings) < self.config.minimum_baseline_samples:
            return None
        temps = [r.temperature for r in readings]
        return float(statistics.median(temps))

    def get_variability(self, sensor_id: str) -> float:
        """Compute standard deviation of recent readings with minimum variability floor."""
        readings = self.history.recent(sensor_id, self.config.baseline_window_size)
        if len(readings) < 2:
            return self.config.minimum_variability
        temps = [r.temperature for r in readings]
        stdev = float(statistics.stdev(temps))
        return max(stdev, self.config.minimum_variability)

    def compute_anomaly_score(self, normalized_dev: float, rate_dev: float = 0.0) -> float:
        """Compute a deterministic, explainable anomaly score in range [0.0, 100.0].

        Args:
            normalized_dev: Standardized temperature deviation (z-score).
            rate_dev: Deviation of temperature rate of change from historical average rate.

        Returns:
            float: Score bounded to [0.0, 100.0].
        """
        sens_key = self.config.anomaly_sensitivity.upper()
        sensitivity_factor = self._SENSITIVITY_MAP.get(sens_key, 1.0)

        eff_z = abs(normalized_dev) * sensitivity_factor
        # Temperature deviation score component (z >= 1.5 maps to score >= 50.0 at MEDIUM sensitivity)
        temp_score = min(100.0, eff_z * 35.0)

        # Rate deviation score component (5.0 °C/s rate dev maps to 15.0 pts)
        rate_score = min(30.0, abs(rate_dev) * 3.0)

        # Weighted combination
        score = (0.85 * temp_score) + (0.15 * rate_score)
        return float(min(100.0, max(0.0, score)))


    def classify_severity(self, score: float) -> AnomalySeverity:
        """Classify anomaly score into AnomalySeverity levels."""
        if score >= self.config.critical_severity_threshold:
            return AnomalySeverity.CRITICAL
        elif score >= self.config.high_severity_threshold:
            return AnomalySeverity.HIGH
        elif score >= self.config.medium_severity_threshold:
            return AnomalySeverity.MEDIUM
        elif score >= self.config.low_severity_threshold:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.NONE

    def detect(self, data: ProcessedThermalData) -> ThermalAnomalyResult:
        """Evaluate processed thermal data for statistical anomalies.

        Args:
            data: ProcessedThermalData instance from Task 2.

        Returns:
            ThermalAnomalyResult: Result containing baseline, deviation, score, status, severity, and evidence.

        Raises:
            InvalidReadingError: If input data is invalid, non-numeric, NaN, or infinite.
        """
        if not isinstance(data, ProcessedThermalData):
            raise InvalidReadingError(f"Anomaly detector accepts ProcessedThermalData instances, got {type(data).__name__}")

        if not isinstance(data.processed_temperature, (int, float)) or not math.isfinite(data.processed_temperature):
            raise InvalidReadingError(f"Processed temperature must be a finite numeric value, got {data.processed_temperature}")

        if not isinstance(data.raw_reading.temperature, (int, float)) or not math.isfinite(data.raw_reading.temperature):
            raise InvalidReadingError(f"Raw temperature must be a finite numeric value, got {data.raw_reading.temperature}")

        reading = data.raw_reading
        sensor_id = reading.sensor_id
        temp = float(data.processed_temperature)
        raw_temp = float(reading.temperature)
        timestamp = reading.timestamp
        current_rate = float(data.rate_of_change)

        # 1. Add raw reading to history
        self.history.add(reading)

        # 2. Check baseline sample availability
        recent_readings = self.history.recent(sensor_id, self.config.baseline_window_size)
        sample_count = len(recent_readings)

        if sample_count < self.config.minimum_baseline_samples:
            prev_status = self._previous_statuses.get(sensor_id)
            self._previous_statuses[sensor_id] = AnomalyStatus.INSUFFICIENT_DATA
            self._previous_severities[sensor_id] = AnomalySeverity.NONE
            return ThermalAnomalyResult(
                timestamp=timestamp,
                sensor_id=sensor_id,
                current_temperature=temp,
                raw_temperature=raw_temp,
                baseline_temperature=None,
                deviation=0.0,
                absolute_deviation=0.0,
                normalized_deviation=0.0,
                variability=0.0,
                anomaly_score=0.0,
                severity=AnomalySeverity.NONE,
                status=AnomalyStatus.INSUFFICIENT_DATA,
                persistence=AnomalyPersistence.NOT_ANOMALOUS,
                transition=AnomalyTransition.NO_CHANGE,
                rate_of_change=current_rate,
                rate_deviation=0.0,
                reasons=[
                    f"Insufficient baseline data ({sample_count}/{self.config.minimum_baseline_samples} minimum samples collected)."
                ],
                not_evaluable_reasons=[],
            )

        # 3. Calculate baseline and variability
        baseline = float(statistics.median([r.temperature for r in recent_readings]))
        deviation = temp - baseline
        abs_dev = abs(deviation)

        variability = self.get_variability(sensor_id)
        normalized_dev = deviation / variability

        # Historical rate calculation
        if len(recent_readings) >= 3:
            rates = []
            for i in range(1, len(recent_readings)):
                dt = (recent_readings[i].timestamp - recent_readings[i - 1].timestamp).total_seconds()
                if dt > 0:
                    rates.append((recent_readings[i].temperature - recent_readings[i - 1].temperature) / dt)
            mean_rate = statistics.mean(rates) if rates else 0.0
            rate_dev = current_rate - mean_rate
        else:
            rate_dev = 0.0

        # 4. Compute Anomaly Score & Severity
        score = self.compute_anomaly_score(normalized_dev, rate_dev)
        severity = self.classify_severity(score)

        prev_status = self._previous_statuses.get(sensor_id)
        prev_severity = self._previous_severities.get(sensor_id)

        # 5. Evaluate Persistence & Status
        is_anomalous_reading = score >= self.config.anomaly_score_threshold

        if is_anomalous_reading:
            count = self._anomalous_counts.get(sensor_id, 0) + 1
            self._anomalous_counts[sensor_id] = count

            start_time = self._anomalous_start_times.get(sensor_id)
            if start_time is None:
                start_time = timestamp
                self._anomalous_start_times[sensor_id] = start_time

            elapsed = (timestamp - start_time).total_seconds()

            if count >= self.config.persistence_count or elapsed >= self.config.persistence_duration:
                persistence = AnomalyPersistence.PERSISTENT
                status = AnomalyStatus.ANOMALY
            else:
                persistence = AnomalyPersistence.TEMPORARY
                status = AnomalyStatus.POSSIBLE_ANOMALY
        else:
            # Recovery / Normal check with hysteresis
            if score <= self.config.recovery_threshold:
                self._anomalous_counts[sensor_id] = 0
                self._anomalous_start_times[sensor_id] = None
                persistence = AnomalyPersistence.NOT_ANOMALOUS

                if prev_status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY):
                    status = AnomalyStatus.RECOVERED
                else:
                    status = AnomalyStatus.NORMAL
            else:
                # In hysteresis zone (recovery_threshold < score < anomaly_score_threshold)
                count = self._anomalous_counts.get(sensor_id, 0)
                if count > 0:
                    persistence = AnomalyPersistence.TEMPORARY
                    status = prev_status if prev_status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY) else AnomalyStatus.NORMAL
                else:
                    persistence = AnomalyPersistence.NOT_ANOMALOUS
                    status = AnomalyStatus.NORMAL

        # 6. Determine Transition
        if prev_status is None or prev_status == AnomalyStatus.INSUFFICIENT_DATA:
            transition = AnomalyTransition.NO_CHANGE
        elif status == AnomalyStatus.RECOVERED or (prev_status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY) and status == AnomalyStatus.NORMAL):
            transition = AnomalyTransition.ANOMALY_RECOVERED
        elif prev_status == AnomalyStatus.NORMAL and status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY):
            transition = AnomalyTransition.ANOMALY_STARTED
        elif severity != prev_severity and prev_severity is not None:
            prev_rank = self._SEVERITY_RANK[prev_severity]
            curr_rank = self._SEVERITY_RANK[severity]
            if curr_rank > prev_rank:
                transition = AnomalyTransition.ANOMALY_ESCALATED
            elif curr_rank < prev_rank:
                transition = AnomalyTransition.ANOMALY_DE_ESCALATED
            else:
                transition = AnomalyTransition.NO_CHANGE
        else:
            transition = AnomalyTransition.NO_CHANGE

        # 7. Build Explainability Reasons
        reasons: List[str] = []
        if is_anomalous_reading or status in (AnomalyStatus.POSSIBLE_ANOMALY, AnomalyStatus.ANOMALY):
            reasons.append(
                f"Temperature ({temp:.2f}°C) deviates from baseline ({baseline:.2f}°C) by {deviation:+.2f}°C (normalized z-score: {normalized_dev:+.2f})."
            )
            reasons.append(
                f"Anomaly score ({score:.1f}/100) exceeds score threshold ({self.config.anomaly_score_threshold:.1f}). Sensitivity: {self.config.anomaly_sensitivity}."
            )
            if persistence == AnomalyPersistence.PERSISTENT:
                reasons.append(
                    f"Anomalous behavior persisted across {self._anomalous_counts[sensor_id]} consecutive readings."
                )
            elif persistence == AnomalyPersistence.TEMPORARY:
                reasons.append(
                    f"Temporary thermal anomaly detected ({self._anomalous_counts[sensor_id]}/{self.config.persistence_count} readings towards persistence)."
                )

        if transition == AnomalyTransition.ANOMALY_RECOVERED:
            reasons.append(
                f"Anomaly recovered. Temperature score ({score:.1f}) returned below recovery threshold ({self.config.recovery_threshold:.1f})."
            )
        elif transition == AnomalyTransition.ANOMALY_STARTED:
            reasons.append(f"Statistical anomaly started on sensor '{sensor_id}'.")
        elif transition == AnomalyTransition.ANOMALY_ESCALATED:
            reasons.append(f"Anomaly severity escalated from {prev_severity.value if prev_severity else 'NONE'} to {severity.value}.")
        elif transition == AnomalyTransition.ANOMALY_DE_ESCALATED:
            reasons.append(f"Anomaly severity de-escalated from {prev_severity.value if prev_severity else 'NONE'} to {severity.value}.")

        # Update per-sensor tracking state
        self._previous_statuses[sensor_id] = status
        self._previous_severities[sensor_id] = severity

        return ThermalAnomalyResult(
            timestamp=timestamp,
            sensor_id=sensor_id,
            current_temperature=temp,
            raw_temperature=raw_temp,
            baseline_temperature=baseline,
            deviation=deviation,
            absolute_deviation=abs_dev,
            normalized_deviation=normalized_dev,
            variability=variability,
            anomaly_score=score,
            severity=severity,
            status=status,
            persistence=persistence,
            transition=transition,
            rate_of_change=current_rate,
            rate_deviation=rate_dev,
            reasons=reasons,
            not_evaluable_reasons=[],
        )

    def reset(self, sensor_id: Optional[str] = None) -> None:
        """Clear internal tracking state and history for a specific sensor or all sensors.

        Args:
            sensor_id: Specific sensor ID string to reset, or None to reset all sensors.
        """
        self.history.clear(sensor_id)
        if sensor_id is None:
            self._previous_statuses.clear()
            self._previous_severities.clear()
            self._anomalous_counts.clear()
            self._anomalous_start_times.clear()
        else:
            self._previous_statuses.pop(sensor_id, None)
            self._previous_severities.pop(sensor_id, None)
            self._anomalous_counts.pop(sensor_id, None)
            self._anomalous_start_times.pop(sensor_id, None)
