import numpy
from matplotlib import pyplot
import warnings

from p1 import divided_differences


"""
Implement Mueller’s method. Use Mueller’s method to find all the roots (real
or complex) of the polynomial
    p(x) = x^4 + x + 1.
"""

# Always
numpy.set_printoptions(suppress=True, precision=4)

# Whether to include visualizing plots
PLOT = True

# Lay out the function and starting samples
FUNCTION = lambda x: x**4 + x + 1
TITLE = "x^4+x+1"
# FUNCTION = lambda x: numpy.cos(x) + 0.25
# TITLE = "cos(x) + 0.25"
SAMPLES = [
    numpy.array([-1.6, -1.5, -1.4]),
    numpy.array([1.6, 1.5, 1.4]),
]

# Optional look at the given function
PLOT_FUNCTION = False
if PLOT_FUNCTION:
    PLOT_X = numpy.linspace(-1.5, 1, 1000)
    pyplot.plot(PLOT_X, [FUNCTION(x) for x in PLOT_X])
    pyplot.show()


def main():
    for X in SAMPLES:
        root, history = mueller_root(FUNCTION, X)

        # Display everything
        print("-" * 80)
        print(f"start values {X} found root {root}")
        if root is not None:
            print(f"evaluated: {FUNCTION(root)}")
        print(f"Took {len(history)} steps")
        print("-" * 80)
        if PLOT and root is not None:

            # Base your plot on a real linspace (this caused issues)
            min_x, max_x = (numpy.min(numpy.real(history)),
                            numpy.max(numpy.real(history)))
            width = max_x - min_x
            min_x -= 0.1 * width
            max_x += 0.1 * width

            figure, axes = pyplot.subplots(2)

            plot_x = numpy.linspace(min_x, max_x, 200)
            plot_y = numpy.array([FUNCTION(x) for x in plot_x])
            data_y = numpy.array([FUNCTION(x) for x in history])
            axes[0].plot([min_x, max_x], [0, 0], 'k--', label="zero")
            axes[0].plot(plot_x, plot_y, 'b', lw=3, label=f"{TITLE}")
            axes[0].plot(history, data_y, 'go', ms=8, label="step history")
            axes[0].plot(history[-1], data_y[-1], 'ro', ms=12, label="final")
            axes[0].set_xlim(min_x, max_x)
            axes[0].legend()
            axes[0].set_title(f"Function y={TITLE}, found root: {root:.3f}")

            plot_x = numpy.linspace(min_x-2, max_x+2, 200)
            plot_y = numpy.array([FUNCTION(x) for x in plot_x])
            min_y, max_y = (numpy.min(plot_y), numpy.max(plot_y))
            axes[1].plot(plot_x, plot_y, 'b', lw=3, label=f"{TITLE}")
            axes[1].plot([min_x,] * 2, [min_y, max_y], 'k--', label="area shown above")
            axes[1].plot([max_x,] * 2, [min_y, max_y], 'k--')
            axes[1].legend()
            axes[1].set_xlim(min_x-2, max_x+2)
            axes[1].set_ylim(0, 30)
            # axes[1].set_ylim(-0.75, 1.25)

    pyplot.show()


def mueller_root(function, X, iterations=int(1e4), eps=1e-10):
    history = X.tolist()
    for _ in range(iterations):
        if numpy.isclose(function(history[-1]), 0.0, atol=eps):
            return (history[-1], history)

        a = divided_differences(history[-3:],
                                [function(x) for x in history[-3:]],
                                deriv_cache={})
        b = a * (history[-1] - history[-2]) + \
            divided_differences(history[-2:],
                                [function(x) for x in history[-2:]],
                                deriv_cache={})
        c = function(history[-1])

        # If we have a non-zero a term, use the quadratic formula
        if abs(a) > 1e-10:
            # Include the j term so the sqrt can return complex results
            discriminant = numpy.sqrt(b**2 - 4*a*c + 0j)
            # Throw away the complex part once the roots are found
            # y0_1 = numpy.real((-b + discriminant) / (2 * a))
            # y0_2 = numpy.real((-b - discriminant) / (2 * a))
            y0_1 = (-b + discriminant) / (2 * a)
            y0_2 = (-b - discriminant) / (2 * a)

            # Use the root that keeps us closer to our starter point in real
            # units
            if abs(numpy.real(y0_1)) < abs(numpy.real(y0_2)):
                history.append(history[-1] + y0_1)
            else:
                history.append(history[-1] + y0_2)
        elif b == 0:
            # Give up if we've hit a local max/min with no slope or curvature
            return (None, history)
        else:
            # Otherwise treat it linearly. 0 = by + c, so we have -c/b = y
            y0 = -c / b
            history.append(history[-1] + y0)

    return (None, history)


if __name__ == "__main__":
    main()
