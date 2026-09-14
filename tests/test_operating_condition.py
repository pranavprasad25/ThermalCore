"""Unit tests for CPU OperatingCondition and DVFS mappers."""

import unittest

from src.cpu.dvfs import (
    DiscreteDVFSMapper,
    FixedOperatingConditionMapper,
    LinearDVFSMapper,
)
from src.cpu.exceptions import InvalidOperatingConditionError
from src.cpu.operating_condition import OperatingCondition
from src.power.config import PowerConfig


class TestOperatingCondition(unittest.TestCase):
    """Test OperatingCondition dataclass and DVFS mappers."""

    def test_operating_condition_validation(self) -> None:
        """Test OperatingCondition bounds checking."""
        op = OperatingCondition(workload=0.5, voltage=1.1, frequency=2.5e9)
        self.assertEqual(op.workload, 0.5)
        self.assertEqual(op.voltage, 1.1)
        self.assertEqual(op.frequency, 2.5e9)

        # Invalid workload
        with self.assertRaises(InvalidOperatingConditionError):
            OperatingCondition(workload=-0.1, voltage=1.0, frequency=2.0e9)

        # Invalid voltage
        with self.assertRaises(InvalidOperatingConditionError):
            OperatingCondition(workload=0.5, voltage=-0.5, frequency=2.0e9)

        # Invalid frequency
        with self.assertRaises(InvalidOperatingConditionError):
            OperatingCondition(workload=0.5, voltage=1.0, frequency=0.0)

    def test_linear_dvfs_mapper(self) -> None:
        """Test LinearDVFSMapper linear scaling between min and max operating points."""
        mapper = LinearDVFSMapper(
            min_voltage=0.8,
            max_voltage=1.2,
            min_frequency=1.0e9,
            max_frequency=3.0e9,
        )

        # At workload = 0.0 -> min bounds
        op0 = mapper.map_workload(0.0)
        self.assertEqual(op0.voltage, 0.8)
        self.assertEqual(op0.frequency, 1.0e9)

        # At workload = 1.0 -> max bounds
        op1 = mapper.map_workload(1.0)
        self.assertEqual(op1.voltage, 1.2)
        self.assertEqual(op1.frequency, 3.0e9)

        # At workload = 0.5 -> midpoint
        op_mid = mapper.map_workload(0.5)
        self.assertAlmostEqual(op_mid.voltage, 1.0)
        self.assertAlmostEqual(op_mid.frequency, 2.0e9)

    def test_fixed_operating_condition_mapper(self) -> None:
        """Test FixedOperatingConditionMapper returns constant V and f."""
        mapper = FixedOperatingConditionMapper(voltage=1.05, frequency=2.4e9)
        op_low = mapper.map_workload(0.1)
        op_high = mapper.map_workload(0.9)

        self.assertEqual(op_low.voltage, 1.05)
        self.assertEqual(op_low.frequency, 2.4e9)
        self.assertEqual(op_high.voltage, 1.05)
        self.assertEqual(op_high.frequency, 2.4e9)

    def test_discrete_dvfs_mapper(self) -> None:
        """Test DiscreteDVFSMapper P-state step transitions."""
        p_states = [
            (0.0, 0.8, 1.0e9),
            (0.4, 1.0, 2.0e9),
            (0.8, 1.2, 3.0e9),
        ]
        mapper = DiscreteDVFSMapper(p_states=p_states)

        # Workload 0.2 -> P-state 0
        op1 = mapper.map_workload(0.2)
        self.assertEqual(op1.voltage, 0.8)

        # Workload 0.5 -> P-state 1
        op2 = mapper.map_workload(0.5)
        self.assertEqual(op2.voltage, 1.0)

        # Workload 0.9 -> P-state 2
        op3 = mapper.map_workload(0.9)
        self.assertEqual(op3.voltage, 1.2)

    def test_linear_dvfs_mapper_with_power_config(self) -> None:
        """Test LinearDVFSMapper inheriting limits from PowerConfig."""
        pcfg = PowerConfig(min_voltage=0.6, max_voltage=1.4, min_frequency=5e8, max_frequency=4e9)
        mapper = LinearDVFSMapper(power_config=pcfg)
        self.assertEqual(mapper.min_voltage, 0.6)
        self.assertEqual(mapper.max_voltage, 1.4)
        self.assertEqual(mapper.min_frequency, 5e8)
        self.assertEqual(mapper.max_frequency, 4e9)


if __name__ == "__main__":
    unittest.main()
