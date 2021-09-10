import numpy

from array_to_latex import to_ltx

from p1 import to_tex


A = numpy.array([
    [7, 6, 4],
    [0, 3, 3],
    [7, 3, 1],
])

Bs = [
    numpy.array([5, -3, 8]).reshape(3, 1),
    numpy.array([2, 0, 11]).reshape(3, 1),
]


U, D, VT = numpy.linalg.svd(A, full_matrices=False)
V = VT.T
D = numpy.diag(D)
Dinv = 1 / D
Dinv[numpy.abs(D) < 1e-10] = 0

for b in Bs:

    x = V @ Dinv @ U.T @ b

    print("\\bar{x} &= V \\Sigma^{-1} U^T b \\\\")
    print(f"&=\n{to_tex(V.T)}\n{to_tex(Dinv)}\n{to_tex(U.T)}\n{to_tex(b)}\\\\")
    print(f"&=\n{to_tex(x)}\\\\")
    print("-" * 80)
    print("A \\bar{x} &= \\bar{b} \\\\")
    print(f"&= {to_tex(A @ x)}")
    print("-" * 80)

x = V @ Dinv @ U.T @ Bs[0]
xn = numpy.array([2, -7, 7]).reshape((3, 1))
assert numpy.allclose(A @ x, Bs[0])
assert numpy.allclose(A @ (x + xn), Bs[0])
