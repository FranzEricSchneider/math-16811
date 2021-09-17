import numpy
from matplotlib import pyplot


"""
Implement a procedure that interpolates $f(x)$ based on a divided difference
approach. The procedure should take as input the following parameters:
    x, x0,...,xn, f(x0),...,f(xn)
The procedure should compute an interpolated value for f(x) based on the given
data points. The procedure should use all the data points (xi, f(xi)),
i = 0,...,n, effectively implementing an interpolating polynomial of degree n
(or less, depending on the data).
"""

# Always
numpy.set_printoptions(suppress=True, precision=4)


def q1pc(n):
    """
    Helper function to figure out the X values for part c based on the number
    of data points n (actually n+1 data points, polynomial degree n).
    """
    return numpy.array([((2 * i) / n) - 1 for i in range(n + 1)])


EXAMPLES = [
    # p1 part b
    (
        lambda x: numpy.exp(x**2 / 2),
        numpy.array([0, 1/8, 1/4, 1/2, 3/4, 1]),
        1/3,
        "e^{x^2 / 2}"
    ),
    # p1 part c
    (lambda x: 1 / (1 + 36 * x**2), q1pc(2), 0.06, "1 / (1 + 36 * x**2), n=2"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(4), 0.06, "1 / (1 + 36 * x**2), n=4"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(6), 0.06, "1 / (1 + 36 * x**2), n=6"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(8), 0.06, "1 / (1 + 36 * x**2), n=8"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(10), 0.06, "1 / (1 + 36 * x**2), n=10"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(12), 0.06, "1 / (1 + 36 * x**2), n=12"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(14), 0.06, "1 / (1 + 36 * x**2), n=14"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(16), 0.06, "1 / (1 + 36 * x**2), n=16"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(18), 0.06, "1 / (1 + 36 * x**2), n=18"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(20), 0.06, "1 / (1 + 36 * x**2), n=20"),
    (lambda x: 1 / (1 + 36 * x**2), q1pc(40), 0.06, "1 / (1 + 36 * x**2), n=40"),
    # just a test
    (
        lambda x: numpy.cos(x),
        numpy.array([0, 1/2, 3/4, 1, 1.5, 2.5, 3.5, 5]),
        1/3,
        "cos(x)"
    ),
    # just a test
    (
        lambda x: numpy.cos(x),
        numpy.array([0, 3/4, 3.5, 5]),
        2,
        "cos(x), low N"
    ),
]


# Number of error points to check out in the given range
ERROR_SLICES = 200


# Whether to include visualizing plots
PLOT = True


def main():
    for function, x_values, eval_at, title in EXAMPLES:
        # Store the cache of derivative values. Without this the n=40 example
        # hung very badly for me
        deriv_cache = {}

        # Calculate the interpolation at the requested point
        y_values = numpy.array([function(x) for x in x_values])
        value = poly_interp(eval_at, x_values, y_values, deriv_cache)

        # Calculate the error across the requested range
        error_range = [numpy.min(x_values), numpy.max(x_values)]
        error_x = numpy.linspace(error_range[0], error_range[1], ERROR_SLICES)
        max_error = numpy.max([
            abs(function(x) - poly_interp(x, x_values, y_values, deriv_cache))
            for x in error_x
        ])

        # Display everything
        print("-" * 80)
        print(f"{title}, interpolated value {value}, actual: {function(eval_at)}")
        print(f"\tMax error across x={error_range} was {max_error}")
        print("-" * 80)
        if PLOT:
            plot_x = numpy.linspace(numpy.min(x_values), numpy.max(x_values), 200)
            plot_y = numpy.array([function(x) for x in plot_x])
            interp_y = numpy.array([poly_interp(x, x_values, y_values, deriv_cache)
                                    for x in plot_x])
            pyplot.plot(plot_x, plot_y, 'k', linewidth=5, label="function")
            pyplot.plot(plot_x, interp_y, 'r--', linewidth=2, label="poly_interp")
            pyplot.plot(x_values, y_values, 'bo', markersize=10, label="data")
            pyplot.plot([eval_at], [value], 'go', markersize=14, label=f"({eval_at:.3f}, {value:.3f})")
            pyplot.legend()
            pyplot.title(f"Function {title}, max_error: {max_error:.3f}")
            pyplot.show()


def poly_interp(eval_at, X, Y, deriv_cache):
    """
    Interpolate the given data points using the (a?) Newton method of
    polynomial interpolation. The general method, described in more detail on
    pages 10-13 of the polynomial notes, is
        f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) ...
    where f[] is a recursive difference between (unsorted) data points.

    https://www.cs.cmu.edu/~me/811/notes/Polynomials.pdf

    Arguments:
        eval_at: float, value at which we want to interpolate
        X, Y: 1D lists/arrays of floats, same length, data points
        deriv_cache: dictionary which get_deriv_term stores already-seen
            derivative values in (helpful for very deep recursion)

    Returns: float, best guess at the interpolation
    """
    return numpy.sum([
        get_deriv_term(X[:i+1], Y[:i+1], deriv_cache) * get_product(eval_at, X[:i])
        for i in range(len(X))
    ])


def get_deriv_term(X, Y, deriv_cache):
    """
    Recursive process. Generally speaking f[x0, ..., xk] is equal to
    (f[x1...xn] - f[x0...x_{n-1}]) / (xk - x0). This lends itself very well to
    recursion. Note that f[xi] = f(xi) = Y[i].

    Arguments:
        X, Y: lists/arrays of floats, must be of the same length
        deriv_cache: dictionary to store already-seen derivative values in
            (helpful for very deep recursion)

    Returns: float value, the result of subtraction and division as discussed
        above and as laid out on page 11 of the polynomial notes:
        https://www.cs.cmu.edu/~me/811/notes/Polynomials.pdf
    """

    if len(X) == 1:
        return Y[0]
    else:
        key = cache_key(X)
        try:
            return deriv_cache[key]
        except KeyError:
            fx_1_k = get_deriv_term(X[1:], Y[1:], deriv_cache)
            fx_0_km1 = get_deriv_term(X[:-1], Y[:-1], deriv_cache)
            deriv_cache[key] = (fx_1_k - fx_0_km1) / (X[-1] - X[0])
            return deriv_cache[key]


def cache_key(X):
    """
    Helper function to provide unique (enough) keys based on the given X
    values. We know that X values are unique in the given data. Good enough.
    """
    return ",".join([f"{x:.4f}" for x in X])


def get_product(eval_at, X):
    """Calculates product of (x - X[i]) for all values in X.

    Arguments:
        eval_at: float, value to lead each of the product terms
        X: 1D list/array of values

    Returns: Calculated value of (x - X[0])(x - X[1])...(x - X[n])
    """
    if len(X) == 0:
        return 1.0
    return numpy.product([(eval_at - x) for x in X])


if __name__ == "__main__":
    main()
