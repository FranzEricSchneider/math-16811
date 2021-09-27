import numpy
import pytest

from p1 import divided_differences, get_product, poly_interp


@pytest.mark.parametrize("x, X, Y, expected", (
    # n=2, simple line
    (0, [-1, 1], [-2, 2], 0),
    (0.5, [-1, 1], [-2, 2], 1),
    (4, [-1, 1], [-2, 2], 8),
    # n=2, different line
    (4, [3, 5], [10, 7], 8.5),
    # n=3, at the data points
    (0, [0, 1, -1], [1, 0, 4], 1),
    (1, [0, 1, -1], [1, 0, 4], 0),
    (-1, [0, 1, -1], [1, 0, 4], 4),
    # Using the known equation, (x - 1)^2
    (-3, [0, 1, -1], [1, 0, 4], (-3 - 1)**2),
    (1.5, [0, 1, -1], [1, 0, 4], (1.5 - 1)**2),
    (4.1, [0, 1, -1], [1, 0, 4], (4.1 - 1)**2),
))
def test_poly_interp(x, X, Y, expected):
    assert numpy.isclose(poly_interp(x, X, Y, {}), expected)


@pytest.mark.parametrize("X, Y, expected", (
    # n=1
    ([0], [10], 10),
    ([100], [10], 10),
    # n=2
    ([-1, 1], [2, 8], 6/2),
    ([10, 12], [2, 8], 6/2),
    ([5, 10], [-4, 12], 16/5),
    # n=3
    ([0, 1, -1], [1, 0, 4], 1),
    ([2, 6, -4], [3, 1, 10], 4/60)
))
def test_divided_differences(X, Y, expected):
    assert numpy.isclose(divided_differences(X, Y, {}), expected)


@pytest.mark.parametrize("eval_at, X, expected", (
    # Start with really simple stuff
    (1, [0], 1),
    (1, [0, 0, 0, 0], 1),
    (3, [3, 3, 3], 0),
    (0, [1, 1, 1], -1),
    (0, [1, 1, 1, 1], 1),
    # A little more complex
    (2, [1, 4, 6, 10], -64),
    (5, [1, 2, 3, 4.5], 12),
    (0.5, [0.25, 0.4, 0.1], 0.01),
    # When passed an empty array, should return 1
    (1e6, [], 1.0),
))
def test_get_product(eval_at, X, expected):
    assert numpy.isclose(get_product(eval_at, X), expected)
