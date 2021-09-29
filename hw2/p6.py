import numpy
from matplotlib import pyplot


# Always
numpy.set_printoptions(suppress=True, precision=4)


F0 = lambda x: x**3 - 2*x**2 + x - 2
F1 = lambda x: x**2 + x - 6


def uncol(A, col_idx):
    """Helper function to drop a column."""
    B = A.copy()
    B = numpy.delete(B, col_idx-1, axis=1)
    return B


def main():

    # For part 2:
    Qp = numpy.array([[0, 1, -2, 1, -2],
                      [1, -2, 1, -2, 0],
                      [0, 0, 1, 1, -6],
                      [0, 1, 1, -6, 0]])
    for i, j in ((1, 2), (2, 3), (3, 4), (4, 5)):
        print(
            f"\\frac{{x_{i}}}{{x_{j}}} &= (-1)^{{{i+j}}} * \\frac{{\\det(Q'_{i})}}{{\\det(Q'_{j})}} ="
            f"{-1**(i+j) * numpy.linalg.det(uncol(Qp, i)) / numpy.linalg.det(uncol(Qp, j))} \\\\"
        )

    # For part 3 (optional)
    Q = numpy.array([[0, 1, -2, 1, -2],
                     [1, -2, 1, -2, 0],
                     [0, 0, 1, 1, -6],
                     [0, 1, 1, -6, 0],
                     [1, 1, -6, 0, 0]])
    u, s, vt = numpy.linalg.svd(Q)
    print(u)
    print(s)
    print(vt / vt[4, 4])

    # For part 1:
    plot_x = numpy.linspace(-1, 2.5, 1000)
    plot_y_0 = numpy.array([F0(x) for x in plot_x])
    plot_y_1 = numpy.array([F1(x) for x in plot_x])
    pyplot.plot([-1, 2.5], [0, 0], "k--", label="zero")
    pyplot.plot(plot_x, plot_y_0, lw=2, label="x^3 - 2x^2 + x - 2")
    pyplot.plot(plot_x, plot_y_1, 'r', lw=2, label="x^2 + x - 6")
    pyplot.ylim(-5, 5)
    pyplot.legend()
    pyplot.title("Appears to have a shared zero")
    pyplot.show()


if __name__ == "__main__":
    main()
