"""Time-varying CPU Workload Profile representation and generator utilities."""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from src.workload.exceptions import InvalidWorkloadProfileError


@dataclass(frozen=True)
class WorkloadPoint:
    """Discrete time-workload sample point.

    Attributes:
        time: Time coordinate in seconds (t >= 0).
        workload: Normalized CPU workload/utilization in range [0.0, 1.0].
    """

    time: float
    workload: float

    def __post_init__(self) -> None:
        """Validate single workload point properties."""
        if not isinstance(self.time, (int, float)) or not math.isfinite(self.time) or self.time < 0.0:
            raise InvalidWorkloadProfileError(f"Timestamp must be a non-negative finite float, got {self.time}")

        if not isinstance(self.workload, (int, float)) or not math.isfinite(self.workload):
            raise InvalidWorkloadProfileError(f"Workload must be a finite float, got {self.workload}")

        if not (0.0 <= self.workload <= 1.0):
            raise InvalidWorkloadProfileError(
                f"Workload must be normalized in range [0.0, 1.0], got {self.workload}"
            )

        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "workload", float(self.workload))

    def to_dict(self) -> Dict[str, float]:
        """Convert WorkloadPoint to dictionary representation."""
        return {"time": round(self.time, 4), "workload": round(self.workload, 4)}


@dataclass
class WorkloadPhase:
    """Phase duration and workload value for phase-based profile construction.

    Attributes:
        duration: Duration of this phase in seconds (> 0).
        workload: Normalized CPU workload [0.0, 1.0] applied during this phase.
    """

    duration: float
    workload: float

    def __post_init__(self) -> None:
        """Validate phase parameters."""
        if not isinstance(self.duration, (int, float)) or not math.isfinite(self.duration) or self.duration <= 0.0:
            raise InvalidWorkloadProfileError(f"Phase duration must be positive and finite, got {self.duration}")

        if not isinstance(self.workload, (int, float)) or not math.isfinite(self.workload) or not (0.0 <= self.workload <= 1.0):
            raise InvalidWorkloadProfileError(f"Phase workload must be in [0.0, 1.0], got {self.workload}")

        self.duration = float(self.duration)
        self.workload = float(self.workload)


class WorkloadProfile:
    """Represents a time-varying CPU workload profile.

    Workload is internally represented as normalized utilization:
        0.0 -> 0% workload (idle)
        1.0 -> 100% workload (maximum load)

    Supports step (piecewise constant) or linear interpolation between discrete time points.
    """

    def __init__(
        self,
        points: Optional[Sequence[Union[WorkloadPoint, Tuple[float, float]]]] = None,
        interpolation: str = "step",
    ) -> None:
        """Initialize a WorkloadProfile.

        Args:
            points: Sequence of WorkloadPoint instances or (time, workload) tuples.
            interpolation: Interpolation strategy between points ('step' or 'linear').

        Raises:
            InvalidWorkloadProfileError: If points are invalid or non-monotonic.
        """
        if interpolation.lower() not in ("step", "linear"):
            raise InvalidWorkloadProfileError(
                f"Interpolation mode must be 'step' or 'linear', got '{interpolation}'"
            )

        self._interpolation = interpolation.lower()
        self._points: List[WorkloadPoint] = []

        if points:
            for item in points:
                if isinstance(item, WorkloadPoint):
                    pt = item
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    pt = WorkloadPoint(time=float(item[0]), workload=float(item[1]))
                else:
                    raise InvalidWorkloadProfileError(
                        f"Expected WorkloadPoint or (time, workload) tuple, got {type(item).__name__}"
                    )
                self._add_validated_point(pt)

    def _add_validated_point(self, pt: WorkloadPoint) -> None:
        """Add a WorkloadPoint enforcing monotonic time ordering."""
        if self._points:
            last_time = self._points[-1].time
            if pt.time < last_time:
                raise InvalidWorkloadProfileError(
                    f"Timestamps must be monotonically non-decreasing. Got {pt.time} after {last_time}"
                )
            if math.isclose(pt.time, last_time, abs_tol=1e-9):
                # Replace last point if exact duplicate timestamp
                self._points[-1] = pt
                return

        self._points.append(pt)

    def add_point(self, time: float, workload: float) -> "WorkloadProfile":
        """Add a single timestamped workload point.

        Args:
            time: Time coordinate in seconds (must be >= latest point time).
            workload: Normalized workload in range [0.0, 1.0].

        Returns:
            Self reference for method chaining.
        """
        pt = WorkloadPoint(time=time, workload=workload)
        self._add_validated_point(pt)
        return self

    def add_phase(self, duration: float, workload: float) -> "WorkloadProfile":
        """Append a workload phase of given duration.

        Args:
            duration: Phase duration in seconds (> 0).
            workload: Normalized workload in range [0.0, 1.0].

        Returns:
            Self reference for method chaining.
        """
        phase = WorkloadPhase(duration=duration, workload=workload)
        if not self._points:
            self.add_point(0.0, phase.workload)
            end_time = phase.duration
            self.add_point(end_time, phase.workload)
        else:
            start_time = self._points[-1].time
            self.add_point(start_time, phase.workload)
            end_time = start_time + phase.duration
            self.add_point(end_time, phase.workload)
        return self

    def get_workload(self, time: float) -> float:
        """Query workload utilization at a specific time t.

        Args:
            time: Evaluation timestamp in seconds.

        Returns:
            Normalized workload value in [0.0, 1.0].

        Raises:
            InvalidWorkloadProfileError: If time is negative or profile is empty.
        """
        if not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0.0:
            raise InvalidWorkloadProfileError(f"Query timestamp must be a non-negative finite float, got {time}")

        if not self._points:
            raise InvalidWorkloadProfileError("Cannot query workload from an empty WorkloadProfile")

        t = float(time)

        # Boundary conditions
        if t <= self._points[0].time:
            return self._points[0].workload
        if t >= self._points[-1].time:
            return self._points[-1].workload

        # Search interval
        for i in range(len(self._points) - 1):
            p1 = self._points[i]
            p2 = self._points[i + 1]
            if p1.time <= t < p2.time:
                if self._interpolation == "step":
                    return p1.workload
                else:  # linear
                    dt = p2.time - p1.time
                    if dt <= 0:
                        return p2.workload
                    alpha = (t - p1.time) / dt
                    return p1.workload + alpha * (p2.workload - p1.workload)

        return self._points[-1].workload

    def sample(self, duration: float, timestep: float) -> Tuple[List[float], List[float]]:
        """Sample the workload profile across a time grid from 0 to duration with given timestep.

        Args:
            duration: Total duration to sample in seconds (> 0).
            timestep: Step size in seconds (> 0).

        Returns:
            Tuple of (times_list, workloads_list).
        """
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise InvalidWorkloadProfileError(f"Duration must be a positive finite float, got {duration}")
        if not isinstance(timestep, (int, float)) or not math.isfinite(timestep) or timestep <= 0:
            raise InvalidWorkloadProfileError(f"Timestep must be a positive finite float, got {timestep}")
        if timestep > duration:
            raise InvalidWorkloadProfileError(
                f"Timestep ({timestep}) cannot be greater than simulation duration ({duration})"
            )

        n_steps = int(math.round(duration / timestep))
        times: List[float] = []
        workloads: List[float] = []

        for k in range(n_steps + 1):
            t_k = round(k * timestep, 10)
            if t_k > duration:
                t_k = duration
            times.append(t_k)
            workloads.append(self.get_workload(t_k))

        return times, workloads

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def points(self) -> List[WorkloadPoint]:
        """List of defined WorkloadPoint objects."""
        return list(self._points)

    @property
    def interpolation(self) -> str:
        """Interpolation strategy ('step' or 'linear')."""
        return self._interpolation

    @property
    def duration(self) -> float:
        """Total time duration covered by defined points in seconds."""
        if not self._points:
            return 0.0
        return self._points[-1].time - self._points[0].time

    @property
    def max_workload(self) -> float:
        """Peak workload utilization in profile."""
        if not self._points:
            return 0.0
        return max(pt.workload for pt in self._points)

    @property
    def min_workload(self) -> float:
        """Minimum workload utilization in profile."""
        if not self._points:
            return 0.0
        return min(pt.workload for pt in self._points)

    @property
    def average_workload(self) -> float:
        """Average workload utilization across defined points."""
        if not self._points:
            return 0.0
        return sum(pt.workload for pt in self._points) / len(self._points)

    def is_empty(self) -> bool:
        """Check whether profile has any defined points."""
        return len(self._points) == 0

    # --------------------------------------------------------------------------
    # Factory Constructors & Pattern Generators
    # --------------------------------------------------------------------------

    @classmethod
    def from_constant(cls, workload: float, duration: float = 10.0) -> "WorkloadProfile":
        """Create a profile with constant workload value.

        Args:
            workload: Constant workload value [0.0, 1.0].
            duration: Simulation duration in seconds.
        """
        prof = cls()
        prof.add_point(0.0, workload)
        prof.add_point(duration, workload)
        return prof

    @classmethod
    def from_step(
        cls,
        steps: Sequence[Tuple[float, float]],
        interpolation: str = "step",
    ) -> "WorkloadProfile":
        """Create a profile from explicit (time, workload) step tuples.

        Args:
            steps: Sequence of (timestamp, workload) tuples.
            interpolation: Interpolation mode ('step' or 'linear').
        """
        if not steps:
            raise InvalidWorkloadProfileError("Cannot create WorkloadProfile from empty steps sequence")
        return cls(points=steps, interpolation=interpolation)

    @classmethod
    def from_phases(cls, phases: Sequence[Tuple[float, float]]) -> "WorkloadProfile":
        """Create a profile from sequence of (duration, workload) phase tuples.

        Args:
            phases: Sequence of (duration_seconds, workload) tuples.
        """
        if not phases:
            raise InvalidWorkloadProfileError("Cannot create WorkloadProfile from empty phases sequence")
        prof = cls()
        for duration, workload in phases:
            prof.add_phase(duration=duration, workload=workload)
        return prof

    @classmethod
    def from_points(
        cls,
        points: Sequence[Union[WorkloadPoint, Tuple[float, float]]],
        interpolation: str = "step",
    ) -> "WorkloadProfile":
        """Create a profile from points or tuples sequence."""
        return cls(points=points, interpolation=interpolation)

    @classmethod
    def from_function(
        cls,
        func: Callable[[float], float],
        duration: float,
        dt: float = 0.1,
        interpolation: str = "linear",
    ) -> "WorkloadProfile":
        """Create a profile by sampling a function f(t) -> workload.

        Args:
            func: Callable accepting time t in seconds and returning workload in [0.0, 1.0].
            duration: Total duration in seconds.
            dt: Sampling interval in seconds.
            interpolation: Interpolation mode.
        """
        if not callable(func):
            raise InvalidWorkloadProfileError("Function argument must be callable")
        if duration <= 0:
            raise InvalidWorkloadProfileError("Duration must be positive")
        if dt <= 0:
            raise InvalidWorkloadProfileError("Sampling interval dt must be positive")

        prof = cls(interpolation=interpolation)
        t = 0.0
        while t <= duration + 1e-9:
            w = float(func(t))
            w_clamped = max(0.0, min(1.0, w))
            prof.add_point(round(t, 6), w_clamped)
            t += dt
        return prof

    @classmethod
    def pattern_constant(cls, workload: float = 0.8, duration: float = 10.0) -> "WorkloadProfile":
        """Pattern A: Constant workload (e.g. 80% for full duration)."""
        return cls.from_constant(workload=workload, duration=duration)

    @classmethod
    def pattern_step(
        cls,
        initial_workload: float = 0.2,
        high_workload: float = 0.8,
        final_workload: float = 0.4,
        step_times: Tuple[float, float] = (2.0, 6.0),
        duration: float = 10.0,
    ) -> "WorkloadProfile":
        """Pattern B: Step workload (e.g. 20% -> 80% -> 40%)."""
        if not (0.0 <= step_times[0] < step_times[1] <= duration):
            raise InvalidWorkloadProfileError(f"Invalid step times {step_times} for duration {duration}")
        prof = cls()
        prof.add_point(0.0, initial_workload)
        prof.add_point(step_times[0], high_workload)
        prof.add_point(step_times[1], final_workload)
        prof.add_point(duration, final_workload)
        return prof

    @classmethod
    def pattern_increasing(
        cls,
        start: float = 0.2,
        end: float = 1.0,
        num_steps: int = 5,
        duration: float = 10.0,
    ) -> "WorkloadProfile":
        """Pattern C: Increasing workload (e.g. 20% -> 40% -> 60% -> 80% -> 100%)."""
        if num_steps < 2:
            raise InvalidWorkloadProfileError("num_steps must be at least 2")
        prof = cls()
        dt = duration / (num_steps - 1)
        dw = (end - start) / (num_steps - 1)
        for i in range(num_steps):
            t = i * dt
            w = start + i * dw
            prof.add_point(round(t, 6), round(w, 4))
        return prof

    @classmethod
    def pattern_decreasing(
        cls,
        start: float = 1.0,
        end: float = 0.2,
        num_steps: int = 5,
        duration: float = 10.0,
    ) -> "WorkloadProfile":
        """Pattern D: Decreasing workload (e.g. 100% -> 80% -> 50% -> 20%)."""
        return cls.pattern_increasing(start=start, end=end, num_steps=num_steps, duration=duration)

    @classmethod
    def pattern_mixed(cls, duration: float = 10.0) -> "WorkloadProfile":
        """Pattern E: Mixed workload (e.g. 20% -> 90% -> 40% -> 100% -> 30%)."""
        # 5 phases across duration
        dt = duration / 5.0
        return cls.from_points(
            [
                (0.0, 0.2),
                (dt, 0.9),
                (2 * dt, 0.4),
                (3 * dt, 1.0),
                (4 * dt, 0.3),
                (duration, 0.3),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert WorkloadProfile to dictionary representation."""
        return {
            "interpolation": self._interpolation,
            "duration": self.duration,
            "min_workload": round(self.min_workload, 4),
            "max_workload": round(self.max_workload, 4),
            "points": [pt.to_dict() for pt in self._points],
        }

    def __repr__(self) -> str:
        return f"WorkloadProfile(points={len(self._points)}, duration={self.duration:.2f}s, interpolation='{self._interpolation}')"
