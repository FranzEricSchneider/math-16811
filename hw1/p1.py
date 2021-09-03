import numpy

"""
Implement the P A = LDU decomposition algorithm discussed in class. Do so
yourself (in other words, do not merely use predefined Gaussian elimination
code in MatLab or Python).
Simplifications:
(i) You may assume that the matrix A is square and invertible.
(ii) Do not worry about column interchanges, just row interchanges.
Demonstrate that your implementation works properly on some examples.
"""


EXAMPLES = [
    numpy.array([
        [1, 1, 0],
        [1, 1, 2],
        [4, 2, 3],
    ])
]


def main():
    for A in EXAMPLES:
        P, L, D, U = decompose_lu(A.astype(float))


def decompose_lu(A):
    """TODO."""

    # Initial assertions
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.dtype == float

    # Initialize the matrices
    P = numpy.eye(A.shape[0])
    L = numpy.eye(A.shape[0])
    DU = A.copy()

    # Then start modifying
    while not full_upper_triangular(DU):
        start = DU.copy()

        # TODO: Put algorithm here

        if numpy.allclose(start, DU):
            raise ValueError("We've gotten into a loop with no progress...")


def full_upper_triangular(DU, threshold=1e-10):
    """
    Returns boolean, true if we have an upper triangular matrix with no zeroes
    on the diagonal.
    """
    # Boolean, using "> threshold" instead of "== 0" to deal with float error
    full_diagonal = all(numpy.abs(numpy.diagonal(DU)) > threshold)

    # Get a mask of the lower triangular and check that the values there are 0
    mask = numpy.tril(numpy.ones(DU.shape, dtype=bool), k=-1)
    lower_empty = all(numpy.abs(DU[mask]) < threshold)

    return full_diagonal and lower_empty


def addrow(P, L, DU, idx0, idx1, scale):
    """Add the idx0 row to the idx1 row, with the given scale.

    NOTE: The matrices (PLA) should be dtype float. Int doesn't work if scale
    is a float, and I don't see a need to guard against it. Just don't give it.
    """
    L[idx1, idx0] -= scale
    DU[idx1] += scale * DU[idx0]
    return P, L, DU


def swaprow(P, L, DU, idx0, idx1):
    """Swap the idx0 row with the idx1 row."""
    # For P and DU, we just do a simple swap
    P[[idx0, idx1]] = P[[idx1, idx0]]
    DU[[idx0, idx1]] = DU[[idx1, idx0]]
    # For L it's a little more complicated. We're supposed to swap the rows
    # as if they were swapped from the beginning. I think an easy way to do
    # this is to subtract the original non-zero values, swap, and add back
    L -= numpy.eye(L.shape[0])
    L[[idx0, idx1]] = L[[idx1, idx0]]
    L += numpy.eye(L.shape[0])
    return P, L, DU


if __name__ == "__main__":
    main()
