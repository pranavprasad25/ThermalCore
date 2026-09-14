"""ThermoShift Root Package.

Thermally-aware processor simulation, workload profiling, power estimation, and thermal RC modeling.
"""

from src.cpu import (
    DiscreteDVFSMapper,
    FixedOperatingConditionMapper,
    LinearDVFSMapper,
    OperatingCondition,
    OperatingConditionMapper,
)
from src.power import PowerConfig, PowerEstimator, PowerResult
from src.simulation import (
    SimulationConfig,
    SimulationStepResult,
    SimulationSummary,
    ThermoShiftSimulation,
    ThermoShiftSimulationResult,
)
from src.safety import (
    ThermalSafetyAnalysis,
    ThermalSafetyConfig,
    ThermalSafetyMonitor,
    ThermalSafetyResult,
    ThermalStatus,
)
from src.thermal import ThermalConfig, ThermalModel, ThermalSimulationResult
from src.workload import WorkloadPhase, WorkloadPoint, WorkloadProfile

__all__ = [
    "WorkloadProfile",
    "WorkloadPoint",
    "WorkloadPhase",
    "OperatingCondition",
    "OperatingConditionMapper",
    "LinearDVFSMapper",
    "FixedOperatingConditionMapper",
    "DiscreteDVFSMapper",
    "PowerConfig",
    "PowerEstimator",
    "PowerResult",
    "ThermalConfig",
    "ThermalModel",
    "ThermalSimulationResult",
    "SimulationConfig",
    "ThermoShiftSimulation",
    "ThermoShiftSimulationResult",
    "SimulationStepResult",
    "SimulationSummary",
    "ThermalSafetyConfig",
    "ThermalStatus",
    "ThermalSafetyResult",
    "ThermalSafetyAnalysis",
    "ThermalSafetyMonitor",
]
