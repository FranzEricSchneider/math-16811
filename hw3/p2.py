from matplotlib import pyplot
import numpy


# Always
numpy.set_printoptions(suppress=True, precision=4)


def main(values):
    xvals = numpy.linspace(0, 1, len(values))
    plot_raw(xvals, values)

    # Calculate the SVD (least squares solution) for projection onto the simple
    # polynomials (1, x, x^2, x^3) etc.
    logvals = numpy.log(values)
    polymatrix = construct_matrix(xvals)

    u, s, vt = numpy.linalg.svd(polymatrix)
    sigma_inv = numpy.zeros(polymatrix.T.shape)
    for i, s_value in enumerate(s):
        sigma_inv[i, i] = 1 / s_value

    weights = vt.T @ sigma_inv @ u.T @ logvals
    print(f"Weights:\n{weights}")

    plot_processed(xvals, values, logvals, weights)


def plot_raw(xvals, values):
    figure, axes = pyplot.subplots(1, 2)

    axes[0].plot(xvals, values)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("value")
    axes[0].set_title("Values")
    axes[1].plot(xvals, numpy.log(values))
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("log(value)")
    axes[1].set_title("Log(values)")

    pyplot.show()


def p_function(x, weights):
    return numpy.sum([
        weights[i] * x**i for i in range(len(weights))
    ])


def e_function(x, weights):
    return numpy.exp(p_function(x, weights))


def plot_processed(xvals, values, logvals, weights):
    figure, axes = pyplot.subplots(1, 2)

    axes[0].plot(xvals, values, "bo", label="Original Data", ms=7)
    axes[0].plot(xvals, [e_function(x, weights) for x in xvals], "r", label="Approximation", lw=3)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("value")
    axes[0].set_title("Raw Values (a.k.a. e^p(x))")
    axes[0].legend()
    axes[1].plot(xvals, logvals, "bo", label="Original Data", ms=7)
    axes[1].plot(xvals, [p_function(x, weights) for x in xvals], "r", label="Approximation", lw=3)
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("log(value)")
    axes[1].set_title("Log(values) (a.k.a. p(x))")
    axes[1].legend()

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
    with open("problem2.txt", "r") as fin:
        string = fin.readline()
    values = numpy.array([float(x) for x in string.split()])
    main(values)
