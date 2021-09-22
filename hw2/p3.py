import numpy
from matplotlib import pyplot


"""
Implement Newton’s Method. Consider the following equation:
    x = tan(x)
There are an infinite number of solutions x to this equation. Use Newton’s
method (and any techniques you need to start Newton in regions of convergence)
to find the two solutions on either side of 19.
"""

# Always
numpy.set_printoptions(suppress=True, precision=4)


# Whether to include visualizing plots
PLOT = True


# Tuples of (title, ylim, function, first derivative of function,
# center value) to pass into root finding
EXAMPLES = [
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 17),
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 18),
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 19),
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 20),
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 21),
    ("tan(x)", 50, numpy.tan, lambda x: (1 / numpy.cos(x))**2, 22),
]


def main():
    for title, ylim, f0, f1, split in EXAMPLES:
        root, history = newton_root(f0, f1, split)

        # Display everything
        print("-" * 80)
        print(f"{title}, start {split} found root {root}, evaluated: {f0(root)}")
        print(f"Took {len(history)} steps")
        print("-" * 80)
        if PLOT:
            min_x, max_x = (numpy.min(history), numpy.max(history))
            width = max_x - min_x
            min_x -= 0.1 * width
            max_x += 0.1 * width

            figure, axes = pyplot.subplots(2)

            plot_x = numpy.linspace(min_x, max_x, 200)
            plot_y = numpy.array([f0(x) for x in plot_x])
            data_y = numpy.array([f0(x) for x in history])
            axes[0].plot([min_x, max_x], [0, 0], 'k--', label="zero")
            axes[0].plot(plot_x, plot_y, 'b', lw=3, label=f"{title}")
            axes[0].plot(history, data_y, 'go', ms=8, label="step history")
            axes[0].plot(history[-1], data_y[-1], 'ro', ms=12, label="final")
            axes[0].set_xlim(min_x, max_x)
            axes[0].legend()
            axes[0].set_title(f"Function y={title}, found root: {root:.3f}")

            plot_x = numpy.linspace(min_x-2, max_x+2, 200)
            plot_y = numpy.array([f0(x) for x in plot_x])
            min_y, max_y = (numpy.min(plot_y), numpy.max(plot_y))
            axes[1].plot(plot_x, plot_y, 'b', lw=3, label=f"{title}")
            axes[1].plot([min_x,] * 2, [min_y, max_y], 'k--', label="area shown above")
            axes[1].plot([max_x,] * 2, [min_y, max_y], 'k--')
            axes[1].legend()
            axes[1].set_xlim(min_x-2, max_x+2)
            axes[1].set_ylim(-ylim, ylim)

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
