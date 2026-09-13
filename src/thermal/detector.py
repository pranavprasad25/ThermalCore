"""Deterministic threshold and rule-based thermal detection engine."""

import math
from datetime import datetime
from typing import Dict, List, Optional

from src.thermal.config import ThermalConfig
from src.thermal.detection_result import (
    StateTransition,
    ThermalCondition,
    ThermalDetectionResult,
    ThermalState,
)
from src.thermal.exceptions import InvalidReadingError
from src.thermal.processed_data import ProcessedThermalData


class ThermalDetector:
    """Deterministic rule-based thermal detection engine.

    Evaluates processed thermal data against configured temperature thresholds and rules,
    producing structured ThermalDetectionResult objects containing detected states,
    conditions, state transitions, and detailed human-readable explanations.
    """

    _STATE_SEVERITY = {
        ThermalState.NORMAL: 0,
        ThermalState.WARNING: 1,
        ThermalState.CRITICAL: 2,
    }

    def __init__(self, config: Optional[ThermalConfig] = None):
        """Initialize ThermalDetector with configuration and multi-sensor tracking state."""
        self.config = config if config is not None else ThermalConfig()
        self._previous_states: Dict[str, ThermalState] = {}
        self._sustained_start_times: Dict[str, Optional[datetime]] = {}

    def classify_temperature(self, temp: float) -> ThermalState:
        """Classify a temperature value into NORMAL, WARNING, or CRITICAL state.

        Args:
            temp: Processed/smoothed temperature in °C.

        Returns:
            ThermalState: State classification based on thresholds.
        """
        if temp >= self.config.critical_temperature:
            return ThermalState.CRITICAL
        elif temp >= self.config.warning_temperature:
            return ThermalState.WARNING
        else:
            return ThermalState.NORMAL

    def evaluate_overheating(self, temp: float) -> bool:
        """Check if temperature meets or exceeds overheating threshold."""
        return temp >= self.config.overheating_temperature

    def evaluate_spike(self, rate_of_change: float) -> bool:
        """Check if rate of change meets or exceeds sudden spike threshold."""
        return rate_of_change >= self.config.spike_threshold

    def evaluate_rapid_rise(self, rate_of_change: float) -> bool:
        """Check if rate of change qualifies as a rapid rise (between rapid rise and spike thresholds)."""
        return self.config.rapid_rise_threshold <= rate_of_change < self.config.spike_threshold

    def evaluate_abnormal_cooling(self, rate_of_change: float) -> bool:
        """Check if rate of change meets or exceeds abnormal cooling threshold (negative rate)."""
        return rate_of_change <= -self.config.abnormal_cooling_threshold

    def evaluate_sustained_high(self, sensor_id: str, temp: float, timestamp: datetime) -> bool:
        """Check if temperature has remained at or above sustained high threshold for configured duration.

        Args:
            sensor_id: Identifier of sensor.
            temp: Current temperature in °C.
            timestamp: Timezone-aware timestamp of current reading.

        Returns:
            bool: True if sustained high condition duration is satisfied, False otherwise.
        """
        if temp >= self.config.sustained_high_temperature:
            start_time = self._sustained_start_times.get(sensor_id)
            if start_time is None:
                start_time = timestamp
                self._sustained_start_times[sensor_id] = start_time

            elapsed = (timestamp - start_time).total_seconds()
            return elapsed >= self.config.sustained_high_duration
        else:
            self._sustained_start_times[sensor_id] = None
            return False

    def determine_transition(
        self, current_state: ThermalState, previous_state: Optional[ThermalState]
    ) -> StateTransition:
        """Categorize the transition between previous state and current state.

        Args:
            current_state: Currently evaluated ThermalState.
            previous_state: Previous ThermalState or None if initial reading.

        Returns:
            StateTransition: Classification of state change.
        """
        if previous_state is None:
            return StateTransition.NO_CHANGE

        if previous_state in (ThermalState.WARNING, ThermalState.CRITICAL) and current_state == ThermalState.NORMAL:
            return StateTransition.RECOVERED

        if current_state == previous_state:
            return StateTransition.NO_CHANGE

        curr_sev = self._STATE_SEVERITY[current_state]
        prev_sev = self._STATE_SEVERITY[previous_state]

        if curr_sev > prev_sev:
            return StateTransition.ESCALATED
        elif curr_sev < prev_sev:
            return StateTransition.DE_ESCALATED
        else:
            return StateTransition.NO_CHANGE

    def detect(self, data: ProcessedThermalData) -> ThermalDetectionResult:
        """Evaluate processed thermal data and generate structured ThermalDetectionResult.

        Args:
            data: ProcessedThermalData instance from ThermalDataProcessor.

        Returns:
            ThermalDetectionResult: Result containing state, conditions, transition, and reasons.

        Raises:
            InvalidReadingError: If data is invalid, malformed, non-numeric, or infinite.
        """
        if not isinstance(data, ProcessedThermalData):
            raise InvalidReadingError(f"Detector accepts ProcessedThermalData instances, got {type(data).__name__}")

        if not isinstance(data.processed_temperature, (int, float)) or not math.isfinite(data.processed_temperature):
            raise InvalidReadingError(f"Processed temperature must be a finite numeric value, got {data.processed_temperature}")

        if not isinstance(data.raw_reading.temperature, (int, float)) or not math.isfinite(data.raw_reading.temperature):
            raise InvalidReadingError(f"Raw temperature must be a finite numeric value, got {data.raw_reading.temperature}")

        reading = data.raw_reading
        sensor_id = reading.sensor_id
        temp = float(data.processed_temperature)
        raw_temp = float(reading.temperature)
        timestamp = reading.timestamp
        rate_of_change = float(data.rate_of_change)

        previous_state = self._previous_states.get(sensor_id)
        current_state = self.classify_temperature(temp)
        transition = self.determine_transition(current_state, previous_state)

        conditions: List[ThermalCondition] = []
        reasons: List[str] = []
        not_evaluable_reasons: List[str] = []

        # 1. Overheating Condition
        if self.evaluate_overheating(temp):
            conditions.append(ThermalCondition.OVERHEATING)
            reasons.append(
                f"Temperature {temp:.2f}°C meets or exceeds overheating threshold ({self.config.overheating_temperature:.2f}°C)."
            )

        # 2. Sudden Spike Condition
        if self.evaluate_spike(rate_of_change):
            conditions.append(ThermalCondition.SUDDEN_SPIKE)
            reasons.append(
                f"Rate of temperature change ({rate_of_change:.2f}°C/s) meets or exceeds spike threshold ({self.config.spike_threshold:.2f}°C/s)."
            )

        # 3. Rapid Rise Condition
        if self.evaluate_rapid_rise(rate_of_change):
            conditions.append(ThermalCondition.RAPID_RISE)
            reasons.append(
                f"Rate of temperature change ({rate_of_change:.2f}°C/s) meets or exceeds rapid rise threshold ({self.config.rapid_rise_threshold:.2f}°C/s)."
            )

        # 4. Abnormal Cooling Condition
        if self.evaluate_abnormal_cooling(rate_of_change):
            conditions.append(ThermalCondition.ABNORMAL_COOLING)
            reasons.append(
                f"Cooling rate ({rate_of_change:.2f}°C/s) meets or exceeds abnormal cooling threshold (-{self.config.abnormal_cooling_threshold:.2f}°C/s)."
            )

        # 5. Sustained High Temperature Condition
        if self.evaluate_sustained_high(sensor_id, temp, timestamp):
            conditions.append(ThermalCondition.SUSTAINED_HIGH)
            reasons.append(
                f"Temperature has remained at or above {self.config.sustained_high_temperature:.2f}°C for sustained duration >= {self.config.sustained_high_duration:.2f}s."
            )

        # 6. Recovery Condition
        if transition == StateTransition.RECOVERED:
            conditions.append(ThermalCondition.RECOVERY)
            reasons.append(
                f"Thermal state recovered from {previous_state.value if previous_state else 'elevated state'} to NORMAL."
            )

        # Explanatory reasons for state and transitions
        if current_state == ThermalState.CRITICAL:
            reasons.append(
                f"Temperature {temp:.2f}°C reached or exceeded critical threshold ({self.config.critical_temperature:.2f}°C)."
            )
        elif current_state == ThermalState.WARNING:
            reasons.append(
                f"Temperature {temp:.2f}°C reached or exceeded warning threshold ({self.config.warning_temperature:.2f}°C)."
            )

        if transition == StateTransition.ESCALATED:
            reasons.append(
                f"Thermal state escalated from {previous_state.value if previous_state else 'None'} to {current_state.value}."
            )
        elif transition == StateTransition.DE_ESCALATED:
            reasons.append(
                f"Thermal state de-escalated from {previous_state.value if previous_state else 'None'} to {current_state.value}."
            )

        # Update per-sensor state tracker
        self._previous_states[sensor_id] = current_state

        return ThermalDetectionResult(
            timestamp=timestamp,
            sensor_id=sensor_id,
            temperature=temp,
            raw_temperature=raw_temp,
            state=current_state,
            previous_state=previous_state,
            conditions=conditions,
            transition=transition,
            rate_of_change=rate_of_change,
            reasons=reasons,
            not_evaluable_reasons=not_evaluable_reasons,
        )

    def reset(self, sensor_id: Optional[str] = None) -> None:
        """Clear internal tracking state for a specific sensor or all sensors.

        Args:
            sensor_id: Specific sensor ID to reset, or None to reset all sensors.
        """
        if sensor_id is None:
            self._previous_states.clear()
            self._sustained_start_times.clear()
        else:
            self._previous_states.pop(sensor_id, None)
            self._sustained_start_times.pop(sensor_id, None)
