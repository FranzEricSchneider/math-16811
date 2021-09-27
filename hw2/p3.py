import numpy
from matplotlib import pyplot
import warnings

"""
Implement Newton’s Method. Consider the following equation:
    x = tan(x)
There are an infinite number of solutions x to this equation. Use Newton’s
method (and any techniques you need to start Newton in regions of convergence)
to find the two solutions on either side of 19.
"""

# Always
numpy.set_printoptions(suppress=True, precision=4)

# Trying to find answers on a slope to infinity often leads to numpy warnings
warnings.filterwarnings('ignore')

# Whether to include visualizing plots
PLOT = True


# Tuples of (title, ylim, function, first derivative of function,
# center value) to pass into root finding
TITLE = "tan(x) - x"
YLIM = 50
F0 = lambda x: numpy.tan(x) - x
F1 = lambda x: ((1 / numpy.cos(x))**2) - 1

# Sample the points with the highest slopes, in the theory that low-slope areas
# are really what kick us into outer space
NUM_SAMPLES = 1000
FRAC_USED = 0.05
SAMPLES = numpy.linspace(15, 23, NUM_SAMPLES)
SAMPLE_SCORES = numpy.array([abs(F1(x)) for x in SAMPLES])
SORTED = numpy.argsort(SAMPLE_SCORES)
SAMPLES = SAMPLES[SORTED[int(-FRAC_USED * NUM_SAMPLES):]]
PLOT_SAMPLES = False
if PLOT_SAMPLES:
    PLOT_X = numpy.linspace(15, 23, 200)
    pyplot.plot(PLOT_X, [F0(x) for x in PLOT_X])
    pyplot.plot(SAMPLES, [F0(x) for x in SAMPLES], 'go')
    pyplot.show()


def main():
    for start in SAMPLES:
        root, history = newton_root(F0, F1, start)

        # Display everything
        print(f"{TITLE}, start {start} found root {root}")
        if root is not None:
            print(f"\tevaluated: {F0(root)}")
            print(f"Took {len(history)} steps")
        if PLOT and root is not None:
            min_x, max_x = (numpy.min(history), numpy.max(history))
            width = max_x - min_x
            min_x -= 0.1 * width
            max_x += 0.1 * width

            figure, axes = pyplot.subplots(2)

            plot_x = numpy.linspace(min_x, max_x, 200)
            plot_y = numpy.array([F0(x) for x in plot_x])
            data_y = numpy.array([F0(x) for x in history])
            axes[0].plot([min_x, max_x], [0, 0], 'k--', label="zero")
            axes[0].plot(plot_x, plot_y, 'b', lw=3, label=f"{TITLE}")
            axes[0].plot(history, data_y, 'go', ms=8, label="step history")
            axes[0].plot(history[-1], data_y[-1], 'ro', ms=12, label="final")
            axes[0].set_xlim(min_x, max_x)
            axes[0].legend()
            axes[0].set_title(f"Function y={TITLE}, started {start:.3f}, found root: {root:.3f}")

            plot_x = numpy.linspace(min_x-2, max_x+2, 200)
            plot_y = numpy.array([F0(x) for x in plot_x])
            min_y, max_y = (numpy.min(plot_y), numpy.max(plot_y))
            axes[1].plot(plot_x, plot_y, 'b', lw=3, label=f"{TITLE}")
            axes[1].plot([min_x,] * 2, [min_y, max_y], 'k--', label="area shown above")
            axes[1].plot([max_x,] * 2, [min_y, max_y], 'k--')
            axes[1].legend()
            axes[1].set_xlim(min_x-2, max_x+2)
            axes[1].set_ylim(-YLIM, YLIM)

    pyplot.show()


def newton_root(f0, f1, start, iterations=int(1e4), eps=1e-10):
    history = [start]
    for _ in range(iterations):
        if numpy.isclose(f0(history[-1]), 0.0, atol=eps):
            return (history[-1], history)
        history.append(history[-1] - f0(history[-1]) / f1(history[-1]))
    return (None, history)


if __name__ == "__main__":
    main()
