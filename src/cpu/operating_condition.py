"""CPU Operating Condition data model."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional

from src.cpu.exceptions import InvalidOperatingConditionError


@dataclass(frozen=True)
class OperatingCondition:
    """Represents CPU operating state (voltage, frequency, workload) at a point in time.

    Attributes:
        workload: Normalized CPU workload/utilization in range [0.0, 1.0].
        voltage: Operating supply voltage in Volts (V).
        frequency: Clock operating frequency in Hertz (Hz).
        activity_factor: Optional direct activity factor override.
        metadata: Optional dictionary of additional state attributes.
    """

    workload: float
    voltage: float
    frequency: float
    activity_factor: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate operating condition parameters."""
        if not isinstance(self.workload, (int, float)) or not math.isfinite(self.workload) or not (0.0 <= self.workload <= 1.0):
            raise InvalidOperatingConditionError(
                f"Workload must be a finite float in [0.0, 1.0], got {self.workload}"
            )

        if not isinstance(self.voltage, (int, float)) or not math.isfinite(self.voltage) or self.voltage <= 0.0:
            raise InvalidOperatingConditionError(f"Voltage must be positive and finite, got {self.voltage}")

        if not isinstance(self.frequency, (int, float)) or not math.isfinite(self.frequency) or self.frequency <= 0.0:
            raise InvalidOperatingConditionError(f"Frequency must be positive and finite, got {self.frequency}")

        if self.activity_factor is not None:
            if not isinstance(self.activity_factor, (int, float)) or not math.isfinite(self.activity_factor) or self.activity_factor < 0.0:
                raise InvalidOperatingConditionError(
                    f"Activity factor must be non-negative and finite, got {self.activity_factor}"
                )
            object.__setattr__(self, "activity_factor", float(self.activity_factor))

        object.__setattr__(self, "workload", float(self.workload))
        object.__setattr__(self, "voltage", float(self.voltage))
        object.__setattr__(self, "frequency", float(self.frequency))

    def to_dict(self) -> Dict[str, Any]:
        """Convert OperatingCondition to dictionary representation."""
        return {
            "workload": round(self.workload, 4),
            "voltage": round(self.voltage, 4),
            "frequency_hz": self.frequency,
            "frequency_ghz": round(self.frequency / 1.0e9, 4),
            "activity_factor": round(self.activity_factor, 6) if self.activity_factor is not None else None,
            "metadata": self.metadata,
        }
