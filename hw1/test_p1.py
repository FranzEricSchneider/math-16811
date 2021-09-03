import numpy
import pytest

from p1 import (addrow,
                best_swap,
                decompose_pldu,
                full_upper_triangular,
                row_not_eligible,
                swaprow,
                )


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


@pytest.mark.parametrize("A", (
    # Start with really simple stuff
    numpy.array([[1, 0], [0, 1]]),
    numpy.array([[-1, 10], [0, 1]]),
    numpy.eye(4),
    # Then make some slightly more realistic arrays
    numpy.array([
        [  4,   3,   2,   1],
        [0.5, 1.2, 0.4, -10],
        [  9,   0,   2, 0.5],
        [ -5, 0.3,  15,   4],
    ]),
    numpy.array([
        [  4,   3,   2,   1],
        [  0, 1.2, 0.4, -10],
        [  0,   0,   2, 0.5],
        [  0,   0,   0,   4],
    ]),
    numpy.array([
        [  0,   0,   0,   4],
        [  0,   0,   2, 0.5],
        [  0, 1.2, 0.4, -10],
        [  4,   3,   2,   1],
    ]),
))
def test_decompose_pldu(A):
    P, L, D, U = decompose_pldu(A.astype(float))

    # First, check the overall shape and type
    assert P.shape == A.shape
    assert L.shape == A.shape
    assert D.shape == A.shape
    assert U.shape == A.shape
    assert P.dtype == float
    assert L.dtype == float
    assert D.dtype == float
    assert U.dtype == float

    # Then check that matrices have the right "shape", a.k.a. only have values
    # in the right places

    # P should be only zeros and ones, and should have a single 1 per row
    assert set(P.flatten()) == {0, 1}
    assert numpy.sum(P) == A.shape[0]

    # L should be lower triangular, with ones on the diagonal
    assert numpy.allclose(L, numpy.tril(L))
    assert numpy.allclose(numpy.diag(L), numpy.ones(A.shape[0]))

    # D should be diagonal, values arbitrary
    assert numpy.allclose(D, numpy.diag(numpy.diag(D)))

    # U should be upper triangular, with ones on the diagonal
    assert numpy.allclose(U, numpy.triu(U))
    assert numpy.allclose(numpy.diag(U), numpy.ones(A.shape[0]))

    # Then do a functional check on some arbitrary vectors
    x_list = [
        numpy.array([0, 1, 2, 3, 4, 5, 6][:A.shape[0]]),
        numpy.array([-5, 2, 3.5, -7.1, 3, 5.5][:A.shape[0]]),
        numpy.array([1e2, -1e1, 1, -1e-1, 1e-2, -1e-3][:A.shape[0]]),
    ]
    for x in x_list:
        assert numpy.allclose(
            P @ A @ x,
            L @ D @ U @ x,
        )


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


@pytest.mark.parametrize("DU, active_idx, expected", (
    (
        numpy.array([
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ]),
        0,
        True,
    ),
    (
        numpy.array([
            [8, 5, 1],
            [0, 0, 3],
            [0, 4, 7],
        ]),
        1,
        True,
    ),
    (
        numpy.array([
            [8,  5, 1],
            [0, -4, 7],
            [0,  0, 3],
        ]),
        1,
        False,
    ),
    (
        numpy.array([
            [8, 5, 1],
            [0, 0, 3],
            [0, 6, 0],
        ]),
        2,
        True,
    ),
))
def test_row_not_eligible(DU, active_idx, expected):
    assert row_not_eligible(DU, active_idx) == expected


@pytest.mark.parametrize("DU, active_idx, expected", (
    (
        numpy.array([
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ]),
        0,
        1,
    ),
    (
        numpy.array([
            [0, 4, 9],
            [1, 2, 0],
            [3, 0, 0],
        ]),
        0,
        2,
    ),
    (
        numpy.array([
            [8, 5, 1, 0],
            [0, 0, 3, 3],
            [0, 4, 0, 4],
            [0, 8, 0, 0],
        ]),
        1,
        3,
    ),
    (
        numpy.array([
            [ 1,  1, 0],
            [ 0,  0, 2],
            [ 0, -2, 3],
        ]),
        1,
        2,
    ),
))
def test_best_swap(DU, active_idx, expected):
    assert best_swap(DU, active_idx) == expected


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
