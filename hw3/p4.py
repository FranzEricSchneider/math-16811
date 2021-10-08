from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy


# Always
numpy.set_printoptions(suppress=True, precision=3)


def read_values(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    return numpy.array([[float(x) for x in line.split()]
                        for line in lines])


def p4a(filename):

    xyz_values = read_values(filename)
    plot_raw(xyz_values, filename)

    u, s, vt = numpy.linalg.svd(xyz_values)
    sigma_inv = numpy.zeros(xyz_values.T.shape)
    for i, s_value in enumerate(s):
        sigma_inv[i, i] = 1 / s_value

    # TODO: Explain
    d_vector = -1 * numpy.ones(xyz_values.shape[0])

    # Calculate weights, then add in the forced d=1 setting
    weights = vt.T @ sigma_inv @ u.T @ d_vector
    weights = numpy.hstack((weights, [1]))
    print(f"Weights:\n{weights}")

    plot_processed(xyz_values, filename, weights)


def plot_raw(xyz, filename):
    figure = pyplot.figure()
    axis = pyplot.axes(projection='3d')
    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "bo")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"Raw values from {filename}")
    pyplot.show()


def plane(X, Y, weights):
    """
    We're working with the plane equation ax + by + cz + d = 0. Therefore
        -cz = ax + by + d
        z = -(ax + by + d) / c
    """
    a, b, c, d = weights
    return -(a * X + b * Y + d) / c


def plot_processed(xyz, filename, weights, n=50):
    figure = pyplot.figure()
    axis = pyplot.axes(projection='3d')

    x = numpy.linspace(numpy.min(xyz[:, 0]), numpy.max(xyz[:, 0]), n)
    y = numpy.linspace(numpy.min(xyz[:, 1]), numpy.max(xyz[:, 1]), n)
    X, Y = numpy.meshgrid(x, y)
    Z = plane(X, Y, weights)
    axis.plot(X.flatten(), Y.flatten(), Z.flatten(), "r", label="Fit")

    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "bo", label="Raw")

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"{filename} values compared to plane, (a, b, c, d) = {weights}")
    axis.legend()
    pyplot.show()


def construct_matrix(xvals, order=5):
    """Construct a polynomal matrix of the x values like so:
        [1, x[0], x[0]^2, x[0]^3, ...]
        ...
        [1, x[n], x[n]^2, x[n]^3, ...]
    For the given order of polynomial, where n is the length of xvals.
    """
    return numpy.array([
        [x**i for i in range(order + 1)]
        for x in xvals
    ])


if __name__ == "__main__":
    p4a("clear_table.txt")
