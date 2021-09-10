import numpy

from array_to_latex import to_ltx

from p1 import to_tex


As = [
    numpy.array([
        [7, 6, 1],
        [4, 5, 1],
        [7, 7, 7],
    ]),
    numpy.array([
        [12, 12, 0, 0],
        [3, 0, -2, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ]),
    numpy.array([
        [7, 6, 4],
        [0, 3, 3],
        [7, 3, 1],
    ]),
]


x = numpy.array([101, -numpy.pi, 1e-4, 12]).reshape(4, 1)


for A in As:
    U, D, V = numpy.linalg.svd(A, full_matrices=False)
    D = numpy.diag(D)

    assert numpy.allclose(
        A,
        # Correct the dimensions for multiplication
        U[:, :A.shape[1]] @ D @ V,
    )
    assert numpy.allclose(
        A @ x[:A.shape[1]],
        U[:, :A.shape[1]] @ D @ V @ x[:A.shape[1]],
    )
    if len(set(U.shape)) == 1:
        assert numpy.allclose(
            U @ U.T, numpy.eye(U.shape[0])
        )
    else:
        for i in range(U.shape[1]):
            assert numpy.isclose(numpy.linalg.norm(U[:, i]), 1.0)
        for i in range(U.shape[1] - 1):
            for j in range(i + 1, U.shape[1]):
                assert numpy.isclose(U[:, i] @ U[:, j], 0.0)
    assert numpy.allclose(
        V @ V.T, numpy.eye(V.shape[0])
    )

    print(f"A &= {to_tex(A)} \\\\")
    print(f"U &= {to_tex(U)}\n&\\Sigma = {to_tex(D)}\n&V^T = {to_tex(V)}")
    print("-" * 80)
