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

# Always
numpy.set_printoptions(suppress=True, precision=4)


EXAMPLES = [
    numpy.array([
        [1, 1, 0],
        [1, 1, 2],
        [4, 2, 3],
    ])
]

# Compare against THRESHOLD instead of "== 0" to deal with float error
THRESHOLD = 1e-10

# Whether to include print statements
DEBUG = True


def main():
    for A in EXAMPLES:
        P, L, D, U = decompose_pldu(A.astype(float))
        if DEBUG:
            print(f"Input A:\n{A}")
            print(f"Final P:\n{P}")
            print(f"Final L:\n{L}")
            print(f"Final D:\n{D}")
            print(f"Final U:\n{U}")


def decompose_pldu(A):
    """Take a non-singular square matrix and decompose it into PA = LDU."""

    # Initial assertions
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.dtype == float
    assert numpy.linalg.det(A) != 0

    # Initialize the matrices
    P = numpy.eye(A.shape[0])
    L = numpy.eye(A.shape[0])
    DU = A.copy()

    # Then start modifying
    active_row = 0
    while not full_upper_triangular(DU):
        start = DU.copy()

        # If the current row is not good (0 on the diagonal), swap rows
        if row_not_eligible(DU, active_row):
            best_idx = best_swap(DU, active_row)
            if DEBUG: print(f"Swapping row{active_row} with row{best_idx}. Starting DU:\n{DU}")
            P, L, DU = swaprow(P, L, DU, active_row, best_idx)
            if DEBUG: print(f"Ending DU:\n{DU}")

        # Zero out the column under the diagonal of the active row
        for row in range(active_row + 1, DU.shape[0]):
            if abs(DU[row, active_row]) > THRESHOLD:
                scale = -DU[row, active_row] / DU[active_row, active_row]
                if DEBUG: print(f"Adding {scale} * row{active_row} with row{row}. Starting DU:\n{DU}")
                P, L, DU = addrow(P, L, DU, active_row, row, scale)
                if DEBUG: print(f"Ending DU:\n{DU}")

        # Advance to the next row
        active_row += 1

        # TODO: Test this assertion with a no-op row. Maybe an empty decomp?
        if numpy.allclose(start, DU):
            raise ValueError("We've gotten into a loop with no progress...")

    # Split D and U
    values = numpy.diag(DU)
    D = numpy.diag(values)
    U = DU.copy()
    for row, value in enumerate(values):
        U[row, :] /= value

    return P, L, D, U


def full_upper_triangular(DU):
    """
    Returns boolean, true if we have an upper triangular matrix with no zeroes
    on the diagonal.
    """
    # Boolean, using "> THRESHOLD" instead of "== 0" to deal with float error
    full_diagonal = all(numpy.abs(numpy.diagonal(DU)) > THRESHOLD)

    # Get a mask of the lower triangular and check that the values there are 0
    mask = numpy.tril(numpy.ones(DU.shape, dtype=bool), k=-1)
    lower_empty = all(numpy.abs(DU[mask]) < THRESHOLD)

    return full_diagonal and lower_empty


def row_not_eligible(DU, idx):
    """Return True if the diagonal for the indicated row is 0."""
    return abs(DU[idx, idx]) < THRESHOLD


def best_swap(DU, active_row):
    """Return the best swap row for the indicated row.

    NOTE: I've chosen the best row as "the one with the most zeros past the
    relevant diagonal", since I figure that will reduce chances for requiring
    future swaps without actually reasoning about future swaps. No idea if this
    is a good criteria.
    """
    print(f"DU: {DU}")
    print(f"active_row + 1: {active_row + 1}")
    print(f"DU.shape[0]: {DU.shape[0]}")
    print(f"range(active_row + 1, DU.shape[0]): {list(range(active_row + 1, DU.shape[0]))}")
    print(f"active_row: {active_row}")
    print(f"DU[idx, active_row]: {DU[2, active_row]}")
    candidates = [idx for idx in range(active_row + 1, DU.shape[0])
                  if abs(DU[idx, active_row]) > THRESHOLD]
    assert len(candidates) > 0
    return sorted(
        candidates,
        key=lambda x: sum(numpy.abs(DU[x, active_row:]) < THRESHOLD),
        reverse=True,
    )[0]


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
