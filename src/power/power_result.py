"""Data models for power estimation results."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PowerResult:
    """Immutable data model representing the output of a power estimation step.

    Attributes:
        dynamic_power: Dynamic switching power dissipation in Watts (W).
        static_power: Static leakage power dissipation in Watts (W).
        total_power: Total power dissipation (dynamic + static) in Watts (W).
        workload: Normalized CPU workload/utilization in range [0.0, 1.0].
        voltage: Operating supply voltage in Volts (V).
        frequency: Clock operating frequency in Hertz (Hz).
        activity_factor: Effective switching activity factor (α).
        capacitance: Effective switching capacitance (C_eff) in Farads (F).
        temperature: Operating junction/core temperature in °C if provided, otherwise None.
        timestamp: Time at which the power calculation was evaluated (UTC).
        metadata: Optional dictionary containing additional diagnostic or environmental details.
    """

    dynamic_power: float
    static_power: float
    total_power: float
    workload: float
    voltage: float
    frequency: float
    activity_factor: float
    capacitance: float
    temperature: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce numeric values and set default timestamp if needed."""
        object.__setattr__(self, "dynamic_power", float(self.dynamic_power))
        object.__setattr__(self, "static_power", float(self.static_power))
        object.__setattr__(self, "total_power", float(self.total_power))
        object.__setattr__(self, "workload", float(self.workload))
        object.__setattr__(self, "voltage", float(self.voltage))
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "activity_factor", float(self.activity_factor))
        object.__setattr__(self, "capacitance", float(self.capacitance))
        if self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert PowerResult instance into a serializable dictionary."""
        return {
            "dynamic_power": round(self.dynamic_power, 6),
            "static_power": round(self.static_power, 6),
            "total_power": round(self.total_power, 6),
            "workload": round(self.workload, 4),
            "voltage": round(self.voltage, 4),
            "frequency_hz": self.frequency,
            "frequency_ghz": round(self.frequency / 1.0e9, 4),
            "activity_factor": round(self.activity_factor, 6),
            "capacitance_f": self.capacitance,
            "temperature_c": round(self.temperature, 2) if self.temperature is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
