import numpy

from p1 import to_tex


# Always
numpy.set_printoptions(suppress=True, precision=4)


numpy.random.seed(12345)
N = 300

# Limit the translation range of the generated HTs
TRANSLATE_RANGE = 500
# Limit the value of points along any given axis
POINT_RANGE = 5
# How far from the origin to move the initially generated points
ZERO_OFFSET = 100

# Treat floats lower than this as 0
THRESHOLD = 1e-10


def rx(theta):
    theta *= 2 * numpy.pi
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    return numpy.array([[1, 0,  0],
                        [0, c, -s],
                        [0, s,  c]])


def rz(theta):
    theta *= 2 * numpy.pi
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    return numpy.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])


def make_ht(a, b, c, d, e, f):
    # Create a 3-vector from [-range, range]
    T = random_to_range(numpy.array([a, b, c]), TRANSLATE_RANGE)
    T = T.reshape((3, 1))

    # Apply 3 separate rotations (technically ZXZ)
    R = numpy.eye(3)
    R = rz(f) @ rx(e) @ rz(d) @ R

    return construct_ht(R, T)


def construct_ht(R, T):
    HT = numpy.hstack([R, T])
    HT = numpy.vstack([HT, numpy.array([0, 0, 0, 1])])
    return HT


def random_to_range(random, radius):
    return (random - 0.5) * 2 * radius


def main():

    transforms = [
        make_ht(*numpy.random.random(6).tolist())
        for i in range(1)
    ]
    noise_levels = [1, 10, 50]
    P = random_to_range(numpy.random.random((3, N)), POINT_RANGE)
    P += random_to_range(numpy.random.random((3, 1)), ZERO_OFFSET)
    P = numpy.vstack([P, numpy.ones(N)])

    for true_transform in transforms:
        for noise in noise_levels:
            Q = true_transform @ P
            Q += random_to_range(numpy.random.random(Q.shape), noise)
            my_transform = calculate_transform(P.copy(), Q.copy(), "mine")
            paper_transform = calculate_transform(P.copy(), Q.copy(), "paper")

            print(f"White noise applied to each transformed point, scaled from [-{noise}, {noise}] along each axis")
            print("\\begin{align*}")
            print("&\\text{True Transform} &\\text{My Transform} \\\\")
            print(f"&{to_tex(true_transform)}")
            print(f"&\\approx {to_tex(my_transform)} \\\\")
            print("& &\\text{Paper Transform} \\\\")
            print(f"& &\\approx {to_tex(paper_transform)}")
            print("\\end{align*}")
            print("\\noindent\\makebox[\\linewidth]{\\rule{6.5in}{0.4pt}}")
            print("")


def calculate_transform(P, Q, version):

    # Adjust both sets of points so the COM is at the origin
    center_p = numpy.average(P, axis=1).reshape((4, 1))
    P -= center_p
    center_q = numpy.average(Q, axis=1).reshape((4, 1))
    Q -= center_q

    P_reduced = P[0:3, :]
    Q_reduced = Q[0:3, :]

    if version == "mine":
        R = calculate_my_rotation(P_reduced, Q_reduced)
    elif version == "paper":
        R = calculate_paper_rotation(P_reduced, Q_reduced)
    else:
        raise ValueError("Unexpected value")

    T = center_q[0:3] - R @ center_p[0:3]

    return construct_ht(R, T)


def calculate_my_rotation(P, Q):
    # Get the SVD solution for P
    u, s, vt = numpy.linalg.svd(P)
    s = numpy.diag(s)
    s = numpy.hstack([s, numpy.zeros((3, N-3))])
    v = vt.T

    # Use the pseudo-inverse of P to calculate R
    # We can cancel with a pseudo-inverse because the rows of P are linearly
    # indepedent, and we thus have a right inverse
    # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
    # We know the pseudoinverse is V (1/E) U.T from here:
    # https://www.johndcook.com/blog/2018/05/05/svd/
    sigma_inverse = invert_sigma(s)

    R = Q @ v @ sigma_inverse @ u.T

    # Now take the SVD of R and throw out the scaling components
    u_r, _, vt_r = numpy.linalg.svd(R)
    R = u_r @ vt_r

    return R


def calculate_paper_rotation(P, Q):
    # See the report
    u, s, vt = numpy.linalg.svd(P @ Q.T)
    v = vt.T

    # Set the sign based on the determinant to avoid axis reflection
    d = numpy.eye(3)
    d[2, 2] = numpy.sign(numpy.linalg.det(v @ u.T))

    return v @ d @ u.T


def invert_sigma(sigma):
    inverse = 1 / sigma
    inverse[sigma < THRESHOLD] = 0
    return inverse.T


if __name__ == '__main__':
    main()
