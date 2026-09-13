"""Standard thermal scenario definitions for integration testing and simulation validation."""

from typing import List
from src.thermal.simulation import ThermalScenario, ThermalScenarioPhase


def get_normal_operation_scenario() -> ThermalScenario:
    """Scenario 1: Normal baseline thermal operation."""
    return ThermalScenario(
        name="1. Normal Operation",
        description="Stable temperature operation with minor normal fluctuations around 45°C.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Normal Baseline Phase 1", duration_seconds=5.0, target_temperature=45.0, noise_std=0.2),
            ThermalScenarioPhase(name="Normal Baseline Phase 2", duration_seconds=5.0, target_temperature=45.2, noise_std=0.2),
        ],
        expected_min_health_score=None,
        expected_final_health_status="HEALTHY",
        expected_conditions=[],
    )


def get_gradual_heating_scenario() -> ThermalScenario:
    """Scenario 2: Gradual thermal heating under increasing workload."""
    return ThermalScenario(
        name="2. Gradual Heating",
        description="Gradual thermal rise from 45°C to 75°C across 15 seconds.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=3.0, target_temperature=45.0),
            ThermalScenarioPhase(name="Mild Warmup", duration_seconds=5.0, target_temperature=60.0),
            ThermalScenarioPhase(name="Moderate Heating", duration_seconds=7.0, target_temperature=75.0),
        ],
        expected_min_health_score=80.0,
        expected_final_health_status=None,
        expected_conditions=[],
    )


def get_rapid_rise_scenario() -> ThermalScenario:
    """Scenario 3: Rapid temperature rise exceeding rate thresholds."""
    return ThermalScenario(
        name="3. Rapid Temperature Rise",
        description="Steep thermal jump from 45°C to 78°C causing rapid rise detection.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=3.0, target_temperature=45.0),
            ThermalScenarioPhase(name="Steep Rise", duration_seconds=2.0, target_temperature=80.0),
            ThermalScenarioPhase(name="Sustained Rise", duration_seconds=3.0, target_temperature=82.0),
        ],
        expected_min_health_score=70.0,
        expected_final_health_status=None,
        expected_conditions=["RAPID_RISE"],
    )


def get_sudden_spike_scenario() -> ThermalScenario:
    """Scenario 4: One-off sudden temperature spike with baseline recovery."""
    return ThermalScenario(
        name="4. Sudden Temperature Spike",
        description="Single +30°C transient spike followed by immediate recovery to baseline.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=3.0, target_temperature=45.0),
            ThermalScenarioPhase(name="Transient Spike", duration_seconds=1.0, target_temperature=45.0, spike_offset=30.0),
            ThermalScenarioPhase(name="Post-Spike Baseline", duration_seconds=8.0, target_temperature=45.0),
        ],
        expected_min_health_score=75.0,
        expected_final_health_status="HEALTHY",
        expected_conditions=["SUDDEN_SPIKE"],
    )


def get_overheating_scenario() -> ThermalScenario:
    """Scenario 5: Overheating exceeding warning and critical safety limits."""
    return ThermalScenario(
        name="5. Overheating",
        description="High thermal load reaching 95°C and triggering overheating alerts.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=2.0, target_temperature=45.0),
            ThermalScenarioPhase(name="Thermal Escalation", duration_seconds=3.0, target_temperature=75.0),
            ThermalScenarioPhase(name="Critical Overheating", duration_seconds=6.0, target_temperature=98.0),
        ],
        expected_min_health_score=50.0,
        expected_final_health_status="CRITICAL",
        expected_conditions=["OVERHEATING"],
    )


def get_sustained_high_scenario() -> ThermalScenario:
    """Scenario 6: Sustained high temperature condition."""
    return ThermalScenario(
        name="6. Sustained High Temperature",
        description="Temperature remaining at 85°C continuously for >= 6 seconds.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=2.0, target_temperature=45.0),
            ThermalScenarioPhase(name="Immediate High Step", duration_seconds=10.0, target_temperature=85.0, spike_offset=40.0),
        ],
        expected_min_health_score=70.0,
        expected_final_health_status="WARNING",
        expected_conditions=["SUSTAINED_HIGH"],
    )


def get_abnormal_cooling_scenario() -> ThermalScenario:
    """Scenario 7: Steep abnormal cooling rate."""
    return ThermalScenario(
        name="7. Abnormal Cooling",
        description="Steep thermal drop from 90°C down to 45°C.",
        initial_temperature=90.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="High Temp Initial", duration_seconds=2.0, target_temperature=90.0),
            ThermalScenarioPhase(name="Steep Cooling Drop", duration_seconds=1.5, target_temperature=45.0),
            ThermalScenarioPhase(name="Settled Normal", duration_seconds=12.0, target_temperature=45.0),
        ],
        expected_min_health_score=None,
        expected_final_health_status="HEALTHY",
        expected_conditions=["ABNORMAL_COOLING"],
    )


def get_critical_condition_scenario() -> ThermalScenario:
    """Scenario 8: Severe critical thermal condition combining multiple stress factors."""
    return ThermalScenario(
        name="8. Critical Condition",
        description="Rapid escalation to 98°C triggering CRITICAL state and multiple alerts.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=1.5, target_temperature=45.0),
            ThermalScenarioPhase(name="Critical Overheating Phase", duration_seconds=5.0, target_temperature=98.0, spike_offset=45.0),
        ],
        expected_min_health_score=30.0,
        expected_final_health_status="CRITICAL",
        expected_conditions=["OVERHEATING"],
    )


def get_recovery_scenario() -> ThermalScenario:
    """Scenario 9: Full escalation to CRITICAL followed by full cooling and RECOVERY."""
    return ThermalScenario(
        name="9. Thermal Recovery Lifecycle",
        description="Full cycle: NORMAL -> WARNING -> CRITICAL -> WARNING -> NORMAL -> RECOVERY.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="1. Baseline Normal", duration_seconds=2.0, target_temperature=45.0),
            ThermalScenarioPhase(name="2. Warning Jump", duration_seconds=2.0, target_temperature=75.0),
            ThermalScenarioPhase(name="3. Critical Spike", duration_seconds=4.0, target_temperature=98.0, spike_offset=20.0),
            ThermalScenarioPhase(name="4. Cooling Down", duration_seconds=2.0, target_temperature=72.0),
            ThermalScenarioPhase(name="5. Fully Settled Normal", duration_seconds=15.0, target_temperature=45.0),
        ],
        expected_min_health_score=40.0,
        expected_final_health_status="HEALTHY",
        expected_conditions=["RECOVERY"],
    )




def get_subthreshold_anomaly_scenario() -> ThermalScenario:
    """Scenario 10: Anomaly below Task 3 fixed safety threshold (68°C vs 45°C baseline)."""
    return ThermalScenario(
        name="10. Sub-threshold Anomaly",
        description="Temperature shift to 68°C (Below 70°C Task 3 limit, but statistical anomaly).",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline Build", duration_seconds=4.0, target_temperature=45.0, noise_std=0.1),
            ThermalScenarioPhase(name="Sub-threshold Shift", duration_seconds=4.0, target_temperature=68.0, noise_std=0.1),
        ],
        expected_min_health_score=85.0,
        expected_final_health_status=None,
        expected_conditions=["THERMAL_ANOMALY"],
    )


def get_temporary_noise_scenario() -> ThermalScenario:
    """Scenario 11: Temporary random noise around baseline."""
    return ThermalScenario(
        name="11. Temporary Noise",
        description="Random noise fluctuations (noise_std=1.5°C) around 45°C baseline.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Noisy Operation", duration_seconds=5.0, target_temperature=45.0, noise_std=1.5),
        ],
        expected_min_health_score=None,
        expected_final_health_status="HEALTHY",
        expected_conditions=[],
    )


def get_persistent_anomaly_scenario() -> ThermalScenario:
    """Scenario 12: Persistent thermal anomaly across multiple consecutive samples."""
    return ThermalScenario(
        name="12. Persistent Thermal Anomaly",
        description="Temperature shifting to 85°C and persisting for >= 5 seconds.",
        initial_temperature=45.0,
        sampling_interval=0.5,
        phases=[
            ThermalScenarioPhase(name="Baseline", duration_seconds=4.0, target_temperature=45.0, noise_std=0.1),
            ThermalScenarioPhase(name="Persistent Shift", duration_seconds=5.0, target_temperature=85.0, noise_std=0.1),
        ],
        expected_min_health_score=75.0,
        expected_final_health_status=None,
        expected_conditions=["PERSISTENT_ANOMALY"],
    )


def get_all_standard_scenarios() -> List[ThermalScenario]:
    """Retrieve ordered list of all 12 standard thermal scenario definitions."""
    return [
        get_normal_operation_scenario(),
        get_gradual_heating_scenario(),
        get_rapid_rise_scenario(),
        get_sudden_spike_scenario(),
        get_overheating_scenario(),
        get_sustained_high_scenario(),
        get_abnormal_cooling_scenario(),
        get_critical_condition_scenario(),
        get_recovery_scenario(),
        get_subthreshold_anomaly_scenario(),
        get_temporary_noise_scenario(),
        get_persistent_anomaly_scenario(),
    ]
