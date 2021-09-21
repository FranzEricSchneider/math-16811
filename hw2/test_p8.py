import numpy
import pytest

from p8 import Path, Triangle


# Define a simple set of paths
paths = [
    Path("0.0 0.5 1.0 1.5",   # X points
         "0.0 0.5 1.0 1.5"),  # Y points
    Path("0.0 0.5 1.0 1.5",
         "1.0 1.5 2.0 2.5"),
    Path("1.0 1.5 2.0 2.5",
         "0.0 0.5 1.0 1.5"),
]


class TestTriangle:

    @pytest.mark.parametrize("point, expected", (
        # Inside
        (numpy.array([0.1, 0.1]), True),
        (numpy.array([1e-3, 1 - 1e-2]), True),
        (numpy.array([0.499, 0.499]), True),
        (numpy.array([0.75, 0.2]), True),
        # Outside
        (numpy.array([-0.1, 0.1]), False),
        (numpy.array([2, 0]), False),
        (numpy.array([0.5, 10]), False),
        (numpy.array([0.5, -1e-3]), False),
        (numpy.array([0.501, 0.501]), False),
    ))
    def test_contains(self, point, expected):
        triangle = Triangle(paths)
        assert triangle.contains(point) == expected
