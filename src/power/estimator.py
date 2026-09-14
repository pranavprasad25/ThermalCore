"""Power Estimation Engine for dynamic, static/leakage, and total power modeling."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional

from src.power.config import PowerConfig
from src.power.exceptions import InvalidPowerParameterError, PowerEstimationError
from src.power.power_result import PowerResult


class PowerEstimator:
    """Subsystem for estimating dynamic, static (leakage), and total CPU power consumption.

    Physics Models:
        - Dynamic Power: P_dynamic = α × C_eff × V² × f
            where α is the switching activity factor derived from workload utilization,
            C_eff is effective switching capacitance in Farads,
            V is operating voltage in Volts,
            f is operating clock frequency in Hertz.
        - Static Power: P_static = P_base × f(T) × g(V)
            where P_base is baseline static leakage power,
            f(T) is thermal leakage scaling (Constant, Linear, or Exponential),
            g(V) is optional voltage-dependent leakage scaling.
        - Total Power: P_total = P_dynamic + P_static
    """

    def __init__(self, config: Optional[PowerConfig] = None) -> None:
        """Initialize the Power Estimator with optional configuration.

        Args:
            config: PowerConfig instance containing electrical and physical parameters.
                    If omitted, default PowerConfig is instantiated.
        """
        self.config: PowerConfig = config if config is not None else PowerConfig()

    def calculate_activity_factor(self, workload: float) -> float:
        """Compute the switching activity factor (α) from normalized workload utilization.

        α = base_activity_factor + (workload × activity_factor_scale)

        Args:
            workload: Normalized CPU workload/utilization in range [0.0, 1.0].

        Returns:
            Calculated activity factor α (non-negative float).
        """
        self._validate_workload(workload)
        alpha = self.config.base_activity_factor + (workload * self.config.activity_factor_scale)
        return max(0.0, float(alpha))

    def validate_inputs(
        self,
        workload: float,
        voltage: float,
        frequency: float,
        temperature: Optional[float] = None,
        capacitance: Optional[float] = None,
        activity_factor: Optional[float] = None,
    ) -> None:
        """Validate operating conditions against physical and configured limits.

        Args:
            workload: Normalized workload in range [0.0, 1.0].
            voltage: Operating voltage in Volts.
            frequency: Operating clock frequency in Hz.
            temperature: Junction temperature in °C (optional).
            capacitance: Effective capacitance in Farads (optional).
            activity_factor: Direct activity factor α (optional).

        Raises:
            InvalidPowerParameterError: If any parameter is out of bounds, non-numeric, or non-finite.
        """
        self._validate_workload(workload)
        self._validate_voltage(voltage)
        self._validate_frequency(frequency)

        if capacitance is not None:
            self._validate_capacitance(capacitance)

        if activity_factor is not None:
            self._validate_activity_factor(activity_factor)

        if temperature is not None:
            self._validate_temperature(temperature)

    def estimate_dynamic_power(
        self,
        workload: float,
        voltage: float,
        frequency: float,
        activity_factor: Optional[float] = None,
        capacitance: Optional[float] = None,
    ) -> float:
        """Calculate dynamic power dissipation: P_dyn = α × C × V² × f.

        Args:
            workload: Normalized CPU workload in range [0.0, 1.0].
            voltage: Operating supply voltage in Volts (V).
            frequency: Operating clock frequency in Hertz (Hz).
            activity_factor: Optional override for activity factor α.
            capacitance: Optional override for effective capacitance C in Farads (F).

        Returns:
            Estimated dynamic power in Watts (W).

        Raises:
            InvalidPowerParameterError: If any operating condition is invalid.
        """
        self.validate_inputs(
            workload=workload,
            voltage=voltage,
            frequency=frequency,
            capacitance=capacitance,
            activity_factor=activity_factor,
        )

        alpha = activity_factor if activity_factor is not None else self.calculate_activity_factor(workload)
        c_eff = capacitance if capacitance is not None else self.config.capacitance

        try:
            p_dyn = alpha * c_eff * (voltage ** 2) * frequency
            if not math.isfinite(p_dyn) or p_dyn < 0.0:
                raise PowerEstimationError(f"Computed dynamic power is invalid: {p_dyn} W")
            return float(p_dyn)
        except OverflowError as exc:
            raise PowerEstimationError(f"Dynamic power calculation overflowed: {exc}") from exc

    def estimate_static_power(
        self,
        voltage: Optional[float] = None,
        temperature: Optional[float] = None,
        base_static_power: Optional[float] = None,
    ) -> float:
        """Calculate static/leakage power dissipation.

        Supports constant, temperature-dependent (linear or exponential), and voltage-dependent models.

        Args:
            voltage: Operating voltage in Volts (V), used if voltage-dependent leakage is enabled.
            temperature: Junction temperature in °C, used if temperature-dependent leakage is modeled.
            base_static_power: Optional override for baseline static power in Watts (W).

        Returns:
            Estimated static leakage power in Watts (W).

        Raises:
            InvalidPowerParameterError: If provided parameters are invalid.
        """
        p_base = base_static_power if base_static_power is not None else self.config.base_static_power
        if not isinstance(p_base, (int, float)) or not math.isfinite(p_base) or p_base < 0:
            raise InvalidPowerParameterError(f"Base static power must be non-negative and finite, got {p_base}")

        if voltage is not None:
            self._validate_voltage(voltage)

        if temperature is not None:
            self._validate_temperature(temperature)

        p_static = float(p_base)

        # Apply temperature dependence if temperature is provided and model is active
        if temperature is not None and self.config.temperature_coefficient > 0:
            delta_t = temperature - self.config.reference_temperature
            model = self.config.static_power_model.upper()

            if model == "LINEAR":
                temp_factor = 1.0 + (self.config.temperature_coefficient * delta_t)
                p_static *= max(0.0, temp_factor)
            elif model == "EXPONENTIAL":
                try:
                    temp_factor = math.exp(self.config.temperature_coefficient * delta_t)
                    p_static *= temp_factor
                except OverflowError as exc:
                    raise PowerEstimationError(f"Static power temperature calculation overflowed: {exc}") from exc

        # Apply voltage dependence if configured and voltage is supplied
        if self.config.voltage_dependent_leakage and voltage is not None:
            v_ratio = voltage / self.config.reference_voltage
            p_static *= max(0.0, v_ratio)

        return max(0.0, float(p_static))

    def estimate_total_power(
        self,
        workload: float,
        voltage: float,
        frequency: float,
        temperature: Optional[float] = None,
        activity_factor: Optional[float] = None,
        capacitance: Optional[float] = None,
    ) -> float:
        """Calculate total power consumption: P_total = P_dynamic + P_static.

        Args:
            workload: Normalized workload in range [0.0, 1.0].
            voltage: Operating voltage in Volts (V).
            frequency: Operating clock frequency in Hertz (Hz).
            temperature: Optional junction temperature in °C.
            activity_factor: Optional direct activity factor override.
            capacitance: Optional capacitance override in Farads.

        Returns:
            Total power in Watts (W).
        """
        p_dynamic = self.estimate_dynamic_power(
            workload=workload,
            voltage=voltage,
            frequency=frequency,
            activity_factor=activity_factor,
            capacitance=capacitance,
        )
        p_static = self.estimate_static_power(
            voltage=voltage,
            temperature=temperature,
        )
        return float(p_dynamic + p_static)

    def estimate(
        self,
        workload: float,
        voltage: float,
        frequency: float,
        temperature: Optional[float] = None,
        activity_factor: Optional[float] = None,
        capacitance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> PowerResult:
        """Estimate comprehensive dynamic, static, and total power for a given operating point.

        Args:
            workload: Normalized CPU workload/utilization in range [0.0, 1.0].
            voltage: Supply operating voltage in Volts (V).
            frequency: Clock frequency in Hertz (Hz).
            temperature: Optional core temperature in °C.
            activity_factor: Optional activity factor override.
            capacitance: Optional effective capacitance override in Farads (F).
            metadata: Optional dictionary for tracking external context or tags.
            timestamp: Optional evaluation timestamp (defaults to current UTC time).

        Returns:
            PowerResult containing individual power components, inputs, and serializable format.
        """
        self.validate_inputs(
            workload=workload,
            voltage=voltage,
            frequency=frequency,
            temperature=temperature,
            capacitance=capacitance,
            activity_factor=activity_factor,
        )

        alpha = activity_factor if activity_factor is not None else self.calculate_activity_factor(workload)
        c_eff = capacitance if capacitance is not None else self.config.capacitance

        p_dynamic = self.estimate_dynamic_power(
            workload=workload,
            voltage=voltage,
            frequency=frequency,
            activity_factor=alpha,
            capacitance=c_eff,
        )
        p_static = self.estimate_static_power(
            voltage=voltage,
            temperature=temperature,
        )
        p_total = p_dynamic + p_static

        return PowerResult(
            dynamic_power=p_dynamic,
            static_power=p_static,
            total_power=p_total,
            workload=float(workload),
            voltage=float(voltage),
            frequency=float(frequency),
            activity_factor=float(alpha),
            capacitance=float(c_eff),
            temperature=float(temperature) if temperature is not None else None,
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
            metadata=metadata if metadata is not None else {},
        )

    def estimate_batch(self, operating_points: List[Dict[str, Any]]) -> List[PowerResult]:
        """Estimate power across a list of operating points.

        Args:
            operating_points: List of dicts, each with keys corresponding to estimate() arguments.

        Returns:
            List of PowerResult instances.
        """
        results: List[PowerResult] = []
        for idx, point in enumerate(operating_points):
            try:
                result = self.estimate(
                    workload=point["workload"],
                    voltage=point["voltage"],
                    frequency=point["frequency"],
                    temperature=point.get("temperature"),
                    activity_factor=point.get("activity_factor"),
                    capacitance=point.get("capacitance"),
                    metadata=point.get("metadata"),
                    timestamp=point.get("timestamp"),
                )
                results.append(result)
            except Exception as exc:
                raise PowerEstimationError(f"Error estimating power for point at index {idx}: {exc}") from exc
        return results

    # --------------------------------------------------------------------------
    # Private Input Validators
    # --------------------------------------------------------------------------

    def _validate_workload(self, workload: Any) -> None:
        if not isinstance(workload, (int, float)) or isinstance(workload, bool) or not math.isfinite(workload):
            raise InvalidPowerParameterError(f"Workload must be a finite number, got {workload}")
        if workload < 0.0:
            raise InvalidPowerParameterError(f"Workload cannot be negative, got {workload}")
        if workload > 1.0:
            raise InvalidPowerParameterError(
                f"Workload must be normalized in range [0.0, 1.0], got {workload}. "
                f"If passing percentage (e.g. {workload}%), divide by 100 first."
            )

    def _validate_voltage(self, voltage: Any) -> None:
        if not isinstance(voltage, (int, float)) or isinstance(voltage, bool) or not math.isfinite(voltage):
            raise InvalidPowerParameterError(f"Voltage must be a finite number in Volts, got {voltage}")
        if voltage <= 0.0:
            raise InvalidPowerParameterError(f"Voltage must be strictly positive, got {voltage} V")
        if voltage < self.config.min_voltage or voltage > self.config.max_voltage:
            raise InvalidPowerParameterError(
                f"Voltage {voltage} V is outside configured valid limits [{self.config.min_voltage} V, {self.config.max_voltage} V]"
            )

    def _validate_frequency(self, frequency: Any) -> None:
        if not isinstance(frequency, (int, float)) or isinstance(frequency, bool) or not math.isfinite(frequency):
            raise InvalidPowerParameterError(f"Frequency must be a finite number in Hz, got {frequency}")
        if frequency <= 0.0:
            raise InvalidPowerParameterError(f"Frequency must be strictly positive, got {frequency} Hz")
        if frequency < self.config.min_frequency or frequency > self.config.max_frequency:
            raise InvalidPowerParameterError(
                f"Frequency {frequency} Hz is outside configured valid limits [{self.config.min_frequency} Hz, {self.config.max_frequency} Hz]"
            )

    def _validate_capacitance(self, capacitance: Any) -> None:
        if not isinstance(capacitance, (int, float)) or isinstance(capacitance, bool) or not math.isfinite(capacitance):
            raise InvalidPowerParameterError(f"Capacitance must be a finite number in Farads, got {capacitance}")
        if capacitance <= 0.0:
            raise InvalidPowerParameterError(f"Capacitance must be strictly positive, got {capacitance} F")

    def _validate_activity_factor(self, activity_factor: Any) -> None:
        if not isinstance(activity_factor, (int, float)) or isinstance(activity_factor, bool) or not math.isfinite(activity_factor):
            raise InvalidPowerParameterError(f"Activity factor must be a finite number, got {activity_factor}")
        if activity_factor < 0.0:
            raise InvalidPowerParameterError(f"Activity factor cannot be negative, got {activity_factor}")

    def _validate_temperature(self, temperature: Any) -> None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature):
            raise InvalidPowerParameterError(f"Temperature must be a finite number in °C, got {temperature}")
        if temperature < -50.0 or temperature > 200.0:
            raise InvalidPowerParameterError(
                f"Temperature {temperature} °C is outside physical CPU operating bounds [-50.0 °C, 200.0 °C]"
            )
