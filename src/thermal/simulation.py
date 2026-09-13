"""Thermal scenario simulation, execution, and reporting framework."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random
from typing import Any, Dict, List, Optional

from src.thermal.config import ThermalConfig
from src.thermal.health_monitor import ThermalHealthMonitor
from src.thermal.health_result import ThermalHealthResult
from src.thermal.reading import ThermalReading


@dataclass
class ThermalScenarioPhase:
    """Definition of a distinct phase within a thermal scenario.

    Attributes:
        name: Name string of phase.
        duration_seconds: Duration of phase in seconds.
        target_temperature: Target temperature in °C toward which simulation drifts.
        noise_std: Standard deviation of Gaussian noise in °C.
        spike_offset: Immediate temperature offset added to first sample of phase.
        description: Description of phase intent.
    """

    name: str
    duration_seconds: float
    target_temperature: float
    noise_std: float = 0.0
    spike_offset: float = 0.0
    description: str = ""


@dataclass
class ThermalScenario:
    """Definition of a complete thermal simulation scenario.

    Attributes:
        name: Short descriptive name of scenario.
        description: Detailed explanation of scenario purpose.
        initial_temperature: Starting temperature in °C.
        sampling_interval: Time delta between consecutive samples in seconds.
        sensor_id: Target sensor identifier.
        phases: Ordered list of ThermalScenarioPhase objects.
        expected_min_health_score: Expected minimum health score bound for validation assertion.
        expected_final_health_status: Expected final ThermalHealthStatus string value.
        expected_conditions: List of Task 3 condition strings expected to trigger.
    """

    name: str
    description: str
    initial_temperature: float = 45.0
    sampling_interval: float = 0.5
    sensor_id: str = "simulated_core_0"
    phases: List[ThermalScenarioPhase] = field(default_factory=list)
    expected_min_health_score: Optional[float] = None
    expected_final_health_status: Optional[str] = None
    expected_conditions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThermalScenarioResultStep:
    """Step result output captured during scenario execution."""

    step_index: int
    timestamp: datetime
    raw_temperature: float
    processed_temperature: float
    trend: str
    rate_of_change: float
    thermal_state: str
    conditions: List[str]
    anomaly_status: str
    anomaly_severity: str
    anomaly_score: float
    health_score: float
    health_status: str
    active_alerts: List[str]
    health_result: ThermalHealthResult


@dataclass
class ThermalScenarioReport:
    """Aggregated summary report produced after running a ThermalScenario."""

    scenario_name: str
    sensor_id: str
    total_samples: int
    min_temperature: float
    max_temperature: float
    final_temperature: float
    final_thermal_state: str
    max_anomaly_severity: str
    max_anomaly_score: float
    min_health_score: float
    final_health_score: float
    final_health_status: str
    all_active_alerts: List[str]
    all_conditions_detected: List[str]
    passed: bool
    assertion_messages: List[str] = field(default_factory=list)
    steps: List[ThermalScenarioResultStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ThermalScenarioReport to standardized dictionary output."""
        return {
            "scenario_name": self.scenario_name,
            "sensor_id": self.sensor_id,
            "total_samples": self.total_samples,
            "min_temperature": round(self.min_temperature, 4),
            "max_temperature": round(self.max_temperature, 4),
            "final_temperature": round(self.final_temperature, 4),
            "final_thermal_state": self.final_thermal_state,
            "max_anomaly_severity": self.max_anomaly_severity,
            "max_anomaly_score": round(self.max_anomaly_score, 4),
            "min_health_score": round(self.min_health_score, 4),
            "final_health_score": round(self.final_health_score, 4),
            "final_health_status": self.final_health_status,
            "all_active_alerts": list(self.all_active_alerts),
            "all_conditions_detected": list(self.all_conditions_detected),
            "passed": self.passed,
            "assertion_messages": list(self.assertion_messages),
        }


class ThermalScenarioSimulator:
    """Deterministic scenario-based thermal reading generator."""

    def __init__(self, seed: Optional[int] = 42):
        """Initialize simulator with reproducible seed."""
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def generate_readings(
        self, scenario: ThermalScenario, base_timestamp: Optional[datetime] = None
    ) -> List[ThermalReading]:
        """Generate a sequence of ThermalReading objects based on a ThermalScenario.

        Args:
            scenario: ThermalScenario definition object.
            base_timestamp: Starting datetime (defaults to current UTC time).

        Returns:
            List[ThermalReading]: Ordered list of generated thermal readings.
        """
        # Reset RNG seed for exact reproducible generation per call
        if self.seed is not None:
            self._rng = random.Random(self.seed)

        start_ts = base_timestamp if base_timestamp is not None else datetime.now(timezone.utc)
        readings: List[ThermalReading] = []

        curr_temp = float(scenario.initial_temperature)
        curr_time = start_ts
        tick = 0

        for phase in scenario.phases:
            duration = max(scenario.sampling_interval, float(phase.duration_seconds))
            step_count = max(1, int(round(duration / scenario.sampling_interval)))
            target_temp = float(phase.target_temperature)

            for step in range(step_count):
                tick += 1
                curr_time += timedelta(seconds=scenario.sampling_interval)

                # Thermal drift toward target
                drift = 0.2 * (target_temp - curr_temp)
                curr_temp += drift

                # One-off spike on first sample of phase
                spike = float(phase.spike_offset) if step == 0 else 0.0

                # Gaussian noise
                noise = self._rng.gauss(0, phase.noise_std) if phase.noise_std > 0 else 0.0

                raw_val = curr_temp + spike + noise

                reading = ThermalReading(
                    timestamp=curr_time,
                    sensor_id=scenario.sensor_id,
                    temperature=float(raw_val),
                    unit="C",
                )
                readings.append(reading)

        return readings


class ThermalScenarioRunner:
    """Runner executing ThermalScenario definitions through the real ThermalHealthMonitor pipeline."""

    def __init__(
        self,
        config: Optional[ThermalConfig] = None,
        monitor: Optional[ThermalHealthMonitor] = None,
    ):
        """Initialize runner with configuration or pre-existing monitor."""
        self.config = config if config is not None else ThermalConfig()
        self.monitor = monitor if monitor is not None else ThermalHealthMonitor(self.config)

    def run(self, scenario: ThermalScenario, seed: Optional[int] = 42) -> ThermalScenarioReport:
        """Execute scenario readings through the real Task 1 -> Task 5 pipeline and produce a ThermalScenarioReport.

        Args:
            scenario: ThermalScenario definition object.
            seed: Random seed for deterministic simulation.

        Returns:
            ThermalScenarioReport: Summary report containing step output and behavioral assertions.
        """
        simulator = ThermalScenarioSimulator(seed=seed)
        readings = simulator.generate_readings(scenario)

        # Reset monitor state for clean scenario run
        self.monitor.reset(scenario.sensor_id)

        steps: List[ThermalScenarioResultStep] = []
        all_active_alerts: List[str] = []
        all_conditions: List[str] = []

        severities_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        max_severity_val = 0
        max_severity_name = "NONE"

        for idx, reading in enumerate(readings, start=1):
            # Pass reading through the REAL Task 1 -> Task 5 pipeline
            health_res = self.monitor.update(reading)

            # Extract condition strings from Task 3
            conds = [c.value for c in self.monitor.detector.detect(self.monitor.processor.process(reading)).conditions]

            for a in health_res.active_alerts:
                if a not in all_active_alerts:
                    all_active_alerts.append(a)

            for c in conds:
                if c not in all_conditions:
                    all_conditions.append(c)

            sev_name = health_res.anomaly_severity.value
            sev_val = severities_order.get(sev_name, 0)
            if sev_val > max_severity_val:
                max_severity_val = sev_val
                max_severity_name = sev_name

            step_obj = ThermalScenarioResultStep(
                step_index=idx,
                timestamp=reading.timestamp,
                raw_temperature=reading.temperature,
                processed_temperature=health_res.temperature,
                trend=health_res.trend.value,
                rate_of_change=health_res.rate_of_change,
                thermal_state=health_res.thermal_state.value,
                conditions=conds,
                anomaly_status=health_res.anomaly_status.value,
                anomaly_severity=health_res.anomaly_severity.value,
                anomaly_score=health_res.anomaly_score,
                health_score=health_res.health_score,
                health_status=health_res.health_status.value,
                active_alerts=list(health_res.active_alerts),
                health_result=health_res,
            )
            steps.append(step_obj)

        temps = [s.raw_temperature for s in steps]
        health_scores = [s.health_score for s in steps]
        anomaly_scores = [s.anomaly_score for s in steps]

        min_temp = min(temps) if temps else 0.0
        max_temp = max(temps) if temps else 0.0
        final_temp = temps[-1] if temps else 0.0

        min_health = min(health_scores) if health_scores else 100.0
        final_health = health_scores[-1] if health_scores else 100.0
        final_status = steps[-1].health_status if steps else "UNKNOWN"
        final_state = steps[-1].thermal_state if steps else "NORMAL"
        max_anomaly = max(anomaly_scores) if anomaly_scores else 0.0

        # Evaluate Scenario Assertions
        passed = True
        assertion_msgs: List[str] = []

        if scenario.expected_min_health_score is not None:
            if min_health > scenario.expected_min_health_score:
                passed = False
                assertion_msgs.append(
                    f"Min health score ({min_health:.1f}) did not drop to or below expected bound ({scenario.expected_min_health_score:.1f})."
                )

        if scenario.expected_final_health_status is not None:
            if final_status != scenario.expected_final_health_status:
                passed = False
                assertion_msgs.append(
                    f"Final health status ('{final_status}') did not match expected ('{scenario.expected_final_health_status}')."
                )

        if scenario.expected_conditions:
            for exp_cond in scenario.expected_conditions:
                if exp_cond not in all_conditions and exp_cond not in all_active_alerts:
                    passed = False
                    assertion_msgs.append(f"Expected condition '{exp_cond}' was not triggered during scenario.")

        return ThermalScenarioReport(
            scenario_name=scenario.name,
            sensor_id=scenario.sensor_id,
            total_samples=len(steps),
            min_temperature=min_temp,
            max_temperature=max_temp,
            final_temperature=final_temp,
            final_thermal_state=final_state,
            max_anomaly_severity=max_severity_name,
            max_anomaly_score=max_anomaly,
            min_health_score=min_health,
            final_health_score=final_health,
            final_health_status=final_status,
            all_active_alerts=all_active_alerts,
            all_conditions_detected=all_conditions,
            passed=passed,
            assertion_messages=assertion_msgs,
            steps=steps,
        )
