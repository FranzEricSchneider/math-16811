import numpy


TEST_VECTORS = [
    numpy.array([1, 2, 3, 4]),
    numpy.array([1e2, -1e0, 1e-2, -1e-4]),
    numpy.array([2381, 4.23423, numpy.pi, -2052]),
]


# We know P is identity for these problems
def validate_ldu(A, L, D, U):
    for full_vector in TEST_VECTORS:
        vector = full_vector[:A.shape[0]]
        print(f"A @ vector: {A @ vector}")
        print(f"L @ D @ U @ vector: {L @ D @ U @ vector}")
        assert numpy.allclose(
            A @ vector,
            L @ D @ U @ vector,
        )


def test_matrix_1():
    A = numpy.array([
        [7, 6, 1],
        [4, 5, 1],
        [7, 7, 7],
    ])
    L = numpy.array([
        [1, 0, 0],
        [4/7, 1, 0],
        [1, 7/11, 1],
    ])
    D = numpy.array([
        [7, 0, 0],
        [0, 11/7, 0],
        [0, 0, 63/11],
    ])
    U = numpy.array([
        [1, 6/7, 1/7],
        [0, 1, 3/11],
        [0, 0, 1],
    ])
    validate_ldu(A, L, D, U)


def test_matrix_2():
    A = numpy.array([
        [12, 12, 0, 0],
        [3, 0, -2, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ])
    L = numpy.array([
        [1,     0,    0, 0, 0],
        [1/4,   1,    0, 0, 0],
        [0,  -1/3,    1, 0, 0],
        [0,     0,    0, 1, 0],
        [0,     0, -3/5, 1, 1],
    ])
    D = numpy.array([
        [12, 0,    0, 0, 0],
        [0, -3,    0, 0, 0],
        [0,  0, -5/3, 0, 0],
        [0,  0,    0, 1, 0],
        [0,  0,    0, 0, 1],
    ])
    U = numpy.array([
        [1, 1, 0, 0],
        [0, 1, 2/3, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    validate_ldu(A, L, D, U)


def test_matrix_3():
    A = numpy.array([
        [7, 6, 4],
        [0, 3, 3],
        [7, 3, 1],
    ])
    L = numpy.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, -1, 1],
    ])
    D = numpy.array([
        [7, 0, 0],
        [0, 3, 0],
        [0, 0, 0],
    ])
    U = numpy.array([
        [1, 6/7, 4/7],
        [0, 1, 1],
        [0, 0, 0],
    ])
    validate_ldu(A, L, D, U)
