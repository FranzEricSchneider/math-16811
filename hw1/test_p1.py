import numpy
import pytest

from p1 import addrow, full_upper_triangular, swaprow


@pytest.fixture
def P():
    return numpy.eye(4, dtype=float)


@pytest.fixture
def L():
    return numpy.array([
        [1, 0, 0, 0],
        [3, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)


@pytest.fixture
def DU():
    return numpy.array([
        [1, 4, 3, 6],
        [0, 9, 7, 3],
        [7, 2, 0, 3],
        [8, 3, 1, 6],
    ], dtype=float)


@pytest.mark.parametrize("DU, expected", (
    # Start with really simple stuff
    (numpy.array([[1, 0], [0, 1]]), True),
    (numpy.array([[-1, 10], [0, 1]]), True),
    (numpy.array([[-1, 10], [0, 0]]), False),
    (numpy.array([[1, 1], [1, 1]]), False),
    # Then make some slightly more realistic arrays
    (
        numpy.array([
            [  4,   3,   2,   1],
            [0.5, 1.2, 0.4, -10],
            [  9,   0,   2, 0.5],
            [ -5, 0.3,  15,   4],
        ]),
        False
    ),
    (
        numpy.array([
            [  4,   3,   2,   1],
            [  0, 1.2, 0.4, -10],
            [  0,   0,   2, 0.5],
            [  0,   0,   0,   4],
        ]),
        True
    ),
    (
        numpy.array([
            [  4,   3,   2,   1],
            [  0, 1.2, 0.4, -10],
            [  0,   0,   0, 0.5],
            [  0,   0,   0,   4],
        ]),
        False
    ),
))
def test_full_upper_triangular(DU, expected):
    assert full_upper_triangular(DU) == expected


class TestAddRow:
    def test_int_scale(self, P, L, DU):
        P, L, DU = addrow(P, L, DU, idx0=2, idx1=3, scale=-2)
        assert numpy.allclose(P, numpy.eye(4))
        assert numpy.allclose(L, numpy.array([
            [1, 0, 0, 0],
            [3, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 1],
        ]))
        assert numpy.allclose(DU, numpy.array([
            [1, 4, 3, 6],
            [0, 9, 7, 3],
            [7, 2, 0, 3],
            [-6, -1, 1, 0],
        ]))

    def test_float_scale(self, P, L, DU):
        P, L, DU = addrow(P, L, DU, idx0=1, idx1=0, scale=0.25)
        assert numpy.allclose(P, numpy.eye(4))
        assert numpy.allclose(L, numpy.array([
            [1, -0.25, 0, 0],
            [3, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]))
        assert numpy.allclose(DU, numpy.array([
            [1, 4 + 9/4, 3 + 7/4, 6 + 3/4],
            [0, 9, 7, 3],
            [7, 2, 0, 3],
            [8, 3, 1, 6],
        ]))


def test_swaprow(P, L, DU):
    P, L, DU = swaprow(P, L, DU, idx0=2, idx1=1)
    assert numpy.allclose(P, numpy.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]))
    assert numpy.allclose(L, numpy.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [3, 0, 1, 0],
        [0, 0, 0, 1],
    ]))
    assert numpy.allclose(DU, numpy.array([
        [1, 4, 3, 6],
        [7, 2, 0, 3],
        [0, 9, 7, 3],
        [8, 3, 1, 6],
    ]))
