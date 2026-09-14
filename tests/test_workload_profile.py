"""Unit tests for WorkloadProfile and workload input generation."""

import math
import unittest

from src.workload.exceptions import InvalidWorkloadProfileError
from src.workload.profile import WorkloadPhase, WorkloadPoint, WorkloadProfile


class TestWorkloadProfile(unittest.TestCase):
    """Test WorkloadPoint, WorkloadPhase, and WorkloadProfile methods."""

    def test_workload_point_creation_and_validation(self) -> None:
        """Test WorkloadPoint creation and boundary validation."""
        pt = WorkloadPoint(time=2.5, workload=0.75)
        self.assertEqual(pt.time, 2.5)
        self.assertEqual(pt.workload, 0.75)
        self.assertEqual(pt.to_dict(), {"time": 2.5, "workload": 0.75})

        # Invalid workload < 0
        with self.assertRaises(InvalidWorkloadProfileError):
            WorkloadPoint(time=1.0, workload=-0.1)

        # Invalid workload > 1
        with self.assertRaises(InvalidWorkloadProfileError):
            WorkloadPoint(time=1.0, workload=1.05)

        # Invalid negative time
        with self.assertRaises(InvalidWorkloadProfileError):
            WorkloadPoint(time=-0.5, workload=0.5)

    def test_workload_phase_validation(self) -> None:
        """Test WorkloadPhase duration and workload bounds."""
        phase = WorkloadPhase(duration=5.0, workload=0.6)
        self.assertEqual(phase.duration, 5.0)
        self.assertEqual(phase.workload, 0.6)

        with self.assertRaises(InvalidWorkloadProfileError):
            WorkloadPhase(duration=0.0, workload=0.5)

        with self.assertRaises(InvalidWorkloadProfileError):
            WorkloadPhase(duration=2.0, workload=1.5)

    def test_constant_workload_profile(self) -> None:
        """Test WorkloadProfile.from_constant."""
        prof = WorkloadProfile.from_constant(workload=0.8, duration=10.0)
        self.assertEqual(prof.get_workload(0.0), 0.8)
        self.assertEqual(prof.get_workload(5.0), 0.8)
        self.assertEqual(prof.get_workload(10.0), 0.8)
        self.assertEqual(prof.get_workload(15.0), 0.8)  # Clamp to last
        self.assertEqual(prof.duration, 10.0)
        self.assertEqual(prof.max_workload, 0.8)
        self.assertEqual(prof.min_workload, 0.8)

    def test_step_interpolation_workload(self) -> None:
        """Test step (piecewise constant) workload query behavior."""
        steps = [(0.0, 0.2), (2.0, 0.8), (6.0, 0.4)]
        prof = WorkloadProfile.from_step(steps=steps, interpolation="step")

        self.assertEqual(prof.get_workload(0.0), 0.2)
        self.assertEqual(prof.get_workload(1.0), 0.2)
        self.assertEqual(prof.get_workload(1.99), 0.2)
        self.assertEqual(prof.get_workload(2.0), 0.8)
        self.assertEqual(prof.get_workload(4.0), 0.8)
        self.assertEqual(prof.get_workload(6.0), 0.4)
        self.assertEqual(prof.get_workload(10.0), 0.4)

    def test_linear_interpolation_workload(self) -> None:
        """Test linear interpolation mode between points."""
        points = [(0.0, 0.0), (10.0, 1.0)]
        prof = WorkloadProfile.from_points(points=points, interpolation="linear")

        self.assertEqual(prof.get_workload(0.0), 0.0)
        self.assertEqual(prof.get_workload(5.0), 0.5)
        self.assertEqual(prof.get_workload(10.0), 1.0)

    def test_phase_based_profile_construction(self) -> None:
        """Test WorkloadProfile.from_phases."""
        phases = [(2.0, 0.2), (4.0, 0.9), (3.0, 0.5)]
        prof = WorkloadProfile.from_phases(phases)
        self.assertAlmostEqual(prof.duration, 9.0)
        self.assertEqual(prof.get_workload(0.0), 0.2)
        self.assertEqual(prof.get_workload(1.5), 0.2)
        self.assertEqual(prof.get_workload(3.0), 0.9)
        self.assertEqual(prof.get_workload(7.0), 0.5)

    def test_function_generator_profile(self) -> None:
        """Test WorkloadProfile.from_function."""
        func = lambda t: 0.5 + 0.5 * math.sin(t)
        prof = WorkloadProfile.from_function(func=func, duration=6.28, dt=0.5)
        self.assertGreater(len(prof.points), 10)
        self.assertGreaterEqual(prof.min_workload, 0.0)
        self.assertLessEqual(prof.max_workload, 1.0)

    def test_non_monotonic_timestamps_raise_error(self) -> None:
        """Test that adding points out of order raises InvalidWorkloadProfileError."""
        prof = WorkloadProfile()
        prof.add_point(5.0, 0.5)
        with self.assertRaises(InvalidWorkloadProfileError):
            prof.add_point(2.0, 0.8)

    def test_empty_profile_query_raises_error(self) -> None:
        """Test querying workload on an empty profile raises error."""
        prof = WorkloadProfile()
        with self.assertRaises(InvalidWorkloadProfileError):
            prof.get_workload(1.0)


if __name__ == "__main__":
    unittest.main()
