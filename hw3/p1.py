from matplotlib import pyplot
import numpy

from hw2_p5 import mueller_root


def main():
    plot_x = numpy.linspace(0, numpy.pi, 1000)
    plot_y = numpy.cos(plot_x)
    pyplot.plot(plot_x, plot_y)
    pyplot.title("cos(x)")
    pyplot.show()

    plot_y = numpy.sin(plot_x)
    pyplot.plot(plot_x, plot_y)
    pyplot.title("sin(x)")
    pyplot.show()

    ROOT = 0.8104702831753705
    pyplot.plot([0, numpy.pi], [0, 0], "k--")
    x2_function = lambda x: numpy.cos(x) + numpy.sin(x) * (x - numpy.pi) + 1
    plot_y = [x2_function(x) for x in plot_x]
    pyplot.plot(plot_x, plot_y)
    pyplot.plot([ROOT], [0], "go")
    pyplot.title("x_2 graph")
    pyplot.show()

    X = numpy.array([0.3, 0.4, 0.5])
    root, history = mueller_root(x2_function, X=X)
    print(f"Root for {X}: {root}, error: {x2_function(root)}")

    print("")
    print(f"E = {(numpy.sin(ROOT) * numpy.pi - 2) / 2}")
    E = 0.13821685286633079

    print("")
    print(f"m = {-numpy.sin(ROOT)} (should also equal {-2 * (1 + E) / numpy.pi})")

    m = -0.7246113537767084
    b = 1.13821685286633079
    plot_y = numpy.cos(plot_x)
    pyplot.plot(plot_x, plot_y, label="cos(x)", lw=3)
    plot_y = m * plot_x + b
    pyplot.plot(plot_x, plot_y, "k", label="mx+b", lw=3)
    for x, sign in ((0, -1), (ROOT, 1), (numpy.pi - ROOT, -1), (numpy.pi, 1)):
        on_line = m * x + b
        pyplot.plot([x, x], [on_line, on_line + sign * E], "r", lw=3)
    pyplot.plot([0], [1], "r", label="E=0.138217", lw=3)
    pyplot.title("cos(x), approximated by line")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
