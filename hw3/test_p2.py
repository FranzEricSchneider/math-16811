import numpy

from p2 import construct_matrix


def test_construct_matrix():
    xvals = numpy.array([1, 2, 3, 4, 5])
    matrix = construct_matrix(xvals, order=3)
    assert numpy.allclose(
        matrix,
        numpy.array([
            [1, 1, 1, 1],
            [1, 2, 4, 8],
            [1, 3, 9, 27],
            [1, 4, 16, 64],
            [1, 5, 25, 125],
        ])
    )
