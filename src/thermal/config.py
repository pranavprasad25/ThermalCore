"""Centralized configuration for power, thermal model, and acquisition parameters."""

from dataclasses import dataclass


@dataclass
class ThermalConfig:
    """Centralized configuration for sensorless power, thermal model, and acquisition parameters.

    Attributes:
        ambient_temperature: Baseline surrounding environment temperature in °C.
        initial_temperature: Starting temperature of CPU cores in °C.
        idle_power: Static baseline power consumption when utilization is 0, in Watts.
        dynamic_power_max: Maximum dynamic power at 100% utilization and reference frequency, in Watts.
        reference_frequency: Baseline reference clock frequency in GHz.
        thermal_resistance: Equivalent thermal resistance R_th between CPU silicon junction and ambient, in °C/W.
        thermal_capacitance: Equivalent thermal capacitance C_th of the core/package, in J/°C.
        simulation_timestep: Discrete simulation tick duration Δt, in seconds.
        min_valid_temp: Minimum valid physical temperature bound in °C.
        max_valid_temp: Maximum valid physical temperature bound in °C.
    """

    ambient_temperature: float = 25.0
    initial_temperature: float = 25.0
    idle_power: float = 5.0
    dynamic_power_max: float = 40.0
    reference_frequency: float = 2.5
    thermal_resistance: float = 1.2
    thermal_capacitance: float = 20.0
    simulation_timestep: float = 0.5
    min_valid_temp: float = -50.0
    max_valid_temp: float = 150.0
    history_capacity: int = 100
    moving_average_window: int = 5
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.3
    trend_tolerance: float = 0.05
    warning_temperature: float = 70.0
    critical_temperature: float = 90.0
    overheating_temperature: float = 90.0
    spike_threshold: float = 10.0
    rapid_rise_threshold: float = 2.0
    abnormal_cooling_threshold: float = 5.0
    sustained_high_temperature: float = 80.0
    sustained_high_duration: float = 5.0
    baseline_window_size: int = 20
    minimum_baseline_samples: int = 5
    anomaly_sensitivity: str = "MEDIUM"
    anomaly_score_threshold: float = 50.0
    low_severity_threshold: float = 20.0
    medium_severity_threshold: float = 50.0
    high_severity_threshold: float = 75.0
    critical_severity_threshold: float = 90.0
    persistence_count: int = 3
    persistence_duration: float = 3.0
    recovery_threshold: float = 15.0
    minimum_variability: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameter physical constraints."""
        import math
        if self.idle_power < 0:
            raise ValueError(f"idle_power must be non-negative, got {self.idle_power}")
        if self.dynamic_power_max < 0:
            raise ValueError(f"dynamic_power_max must be non-negative, got {self.dynamic_power_max}")
        if self.reference_frequency <= 0:
            raise ValueError(f"reference_frequency must be positive, got {self.reference_frequency}")
        if self.thermal_resistance <= 0:
            raise ValueError(f"thermal_resistance must be positive, got {self.thermal_resistance}")
        if self.thermal_capacitance <= 0:
            raise ValueError(f"thermal_capacitance must be positive, got {self.thermal_capacitance}")
        if self.simulation_timestep <= 0:
            raise ValueError(f"simulation_timestep must be positive, got {self.simulation_timestep}")
        if self.min_valid_temp >= self.max_valid_temp:
            raise ValueError(f"min_valid_temp ({self.min_valid_temp}) must be less than max_valid_temp ({self.max_valid_temp})")
        if self.history_capacity <= 0:
            raise ValueError(f"history_capacity must be positive, got {self.history_capacity}")
        if self.moving_average_window <= 0:
            raise ValueError(f"moving_average_window must be positive, got {self.moving_average_window}")
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError(f"smoothing_alpha must be in range (0.0, 1.0], got {self.smoothing_alpha}")
        if self.trend_tolerance < 0:
            raise ValueError(f"trend_tolerance must be non-negative, got {self.trend_tolerance}")

        # Task 3 Threshold Validations
        if not math.isfinite(self.warning_temperature) or not math.isfinite(self.critical_temperature):
            raise ValueError("Thermal thresholds must be finite numbers.")
        if self.warning_temperature >= self.critical_temperature:
            raise ValueError(f"warning_temperature ({self.warning_temperature}) must be strictly less than critical_temperature ({self.critical_temperature})")
        if self.spike_threshold <= 0 or not math.isfinite(self.spike_threshold):
            raise ValueError(f"spike_threshold must be positive and finite, got {self.spike_threshold}")
        if self.rapid_rise_threshold <= 0 or not math.isfinite(self.rapid_rise_threshold):
            raise ValueError(f"rapid_rise_threshold must be positive and finite, got {self.rapid_rise_threshold}")
        if self.abnormal_cooling_threshold <= 0 or not math.isfinite(self.abnormal_cooling_threshold):
            raise ValueError(f"abnormal_cooling_threshold must be positive and finite, got {self.abnormal_cooling_threshold}")
        if self.sustained_high_duration <= 0 or not math.isfinite(self.sustained_high_duration):
            raise ValueError(f"sustained_high_duration must be positive and finite, got {self.sustained_high_duration}")

        # Task 4 Anomaly Detection Validations
        if self.baseline_window_size <= 0:
            raise ValueError(f"baseline_window_size must be positive, got {self.baseline_window_size}")
        if self.minimum_baseline_samples <= 0:
            raise ValueError(f"minimum_baseline_samples must be positive, got {self.minimum_baseline_samples}")
        if self.minimum_baseline_samples > self.baseline_window_size:
            raise ValueError(f"minimum_baseline_samples ({self.minimum_baseline_samples}) cannot exceed baseline_window_size ({self.baseline_window_size})")
        if not isinstance(self.anomaly_sensitivity, str) or self.anomaly_sensitivity.upper() not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"anomaly_sensitivity must be one of ('LOW', 'MEDIUM', 'HIGH'), got '{self.anomaly_sensitivity}'")
        if not math.isfinite(self.anomaly_score_threshold) or not (0.0 <= self.anomaly_score_threshold <= 100.0):
            raise ValueError(f"anomaly_score_threshold must be in range [0.0, 100.0], got {self.anomaly_score_threshold}")
        if not (0.0 <= self.low_severity_threshold <= self.medium_severity_threshold <= self.high_severity_threshold <= self.critical_severity_threshold <= 100.0):
            raise ValueError("Severity thresholds must satisfy 0 <= low <= medium <= high <= critical <= 100")
        if self.persistence_count <= 0:
            raise ValueError(f"persistence_count must be positive, got {self.persistence_count}")
        if self.persistence_duration <= 0 or not math.isfinite(self.persistence_duration):
            raise ValueError(f"persistence_duration must be positive and finite, got {self.persistence_duration}")
        if not math.isfinite(self.recovery_threshold) or not (0.0 <= self.recovery_threshold <= self.anomaly_score_threshold):
            raise ValueError(f"recovery_threshold must be in range [0.0, anomaly_score_threshold], got {self.recovery_threshold}")
        if self.minimum_variability <= 0 or not math.isfinite(self.minimum_variability):
            raise ValueError(f"minimum_variability must be positive and finite, got {self.minimum_variability}")



