"""Thermal subpackage for ThermoShift / ThermalCore."""

from .config import ThermalConfig
from .exceptions import (
    ThermalAcquisitionError,
    InvalidReadingError,
    SensorReadError,
    UnsupportedUnitError,
)
from .reading import ThermalReading, normalize_to_celsius
from .sensor import ThermalSensor, SimulatedThermalSensor
from .acquisition import ThermalDataAcquisition
from .history import ThermalHistoryBuffer
from .processed_data import ProcessedThermalData, ThermalTrend
from .processor import ThermalDataProcessor
from .detection_result import (
    ThermalState,
    ThermalCondition,
    StateTransition,
    ThermalDetectionResult,
)
from .detector import ThermalDetector
from .anomaly_result import (
    AnomalyStatus,
    AnomalySeverity,
    AnomalyPersistence,
    AnomalyTransition,
    ThermalAnomalyResult,
)
from .anomaly_detector import ThermalAnomalyDetector
from .health_result import ThermalHealthStatus, ThermalHealthResult
from .health_monitor import ThermalHealthMonitor
from .simulation import (
    ThermalScenarioPhase,
    ThermalScenario,
    ThermalScenarioResultStep,
    ThermalScenarioReport,
    ThermalScenarioSimulator,
    ThermalScenarioRunner,
)
from .scenarios import get_all_standard_scenarios

__all__ = [
    "ThermalConfig",
    "ThermalAcquisitionError",
    "InvalidReadingError",
    "SensorReadError",
    "UnsupportedUnitError",
    "ThermalReading",
    "normalize_to_celsius",
    "ThermalSensor",
    "SimulatedThermalSensor",
    "ThermalDataAcquisition",
    "ThermalHistoryBuffer",
    "ProcessedThermalData",
    "ThermalTrend",
    "ThermalDataProcessor",
    "ThermalState",
    "ThermalCondition",
    "StateTransition",
    "ThermalDetectionResult",
    "ThermalDetector",
    "AnomalyStatus",
    "AnomalySeverity",
    "AnomalyPersistence",
    "AnomalyTransition",
    "ThermalAnomalyResult",
    "ThermalAnomalyDetector",
    "ThermalHealthStatus",
    "ThermalHealthResult",
    "ThermalHealthMonitor",
    "ThermalScenarioPhase",
    "ThermalScenario",
    "ThermalScenarioResultStep",
    "ThermalScenarioReport",
    "ThermalScenarioSimulator",
    "ThermalScenarioRunner",
    "get_all_standard_scenarios",
]





