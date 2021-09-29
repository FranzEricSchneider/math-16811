import numpy
from matplotlib import pyplot
import sympy

from p5 import mueller_root


def main():
    # For part 1
    draw_zero_contours()
    # For part 2
    det_q()
    x_roots()
    y_roots()
    draw_zero_contours(x_highlights=[-2.084059, -1.297907],
                       y_highlights=[-1.702093, -0.915941])

def draw_zero_contours(x_highlights=None, y_highlights=None):
    """
    Inspired by
    https://www.geeksforgeeks.org/how-to-draw-a-circle-using-matplotlib-in-python/
    """
    xlim = (-4, 1)
    ylim = (-4, 1)
    x = numpy.linspace(xlim[0], xlim[1], 1000)
    y = numpy.linspace(ylim[0], ylim[1], 1000)
    a, b = numpy.meshgrid(x , y)

    p = 2*a**2 + 2*b**2 + 8*a + 4*b + 9
    q = a**2 + b**2 + 2*a*b + 5*a + 7*b + 8

    figure, axes = pyplot.subplots()

    if x_highlights is not None:
        pyplot.plot([0], [0], "r--", label="resultant x roots")
        for x_high in x_highlights:
            pyplot.plot([x_high]*2, ylim, "r--")
    if y_highlights is not None:
        pyplot.plot([0], [0], "k--", label="corresponding y roots")
        for y_high in y_highlights:
            pyplot.plot(xlim, [y_high]*2, "k--")

    axes.plot([0], [0], "b", label="p(x, y)")
    axes.contour(a , b , p , [0], colors=["b"])
    axes.plot([0], [0], "g", label="q(x, y)")
    axes.contour(a , b , q , [0], colors=["g"])
    axes.set_aspect(1)

    pyplot.legend()
    pyplot.title('Zero contours of p(x, y) and q(x, y)')
    pyplot.show()


def det_q():
    x = sympy.Symbol('x')
    Q = sympy.matrices.Matrix([
        [0, 2, 4, (2*x**2 + 8*x + 9)],
        [2, 4, (2*x**2 + 8*x + 9), 0],
        [0, 1, (2*x + 7), (x**2 + 5*x + 8)],
        [1, (2*x + 7), (x**2 + 5*x + 8), 0],
    ])
    print(f"Calculated determinant {Q.det()}")


def x_roots():
    function = lambda x: 16*x**4 + 144*x**3 + 480*x**2 + 692*x + 359
    for i in range(-5, 5):
        x = numpy.array([i-1, i, i+1])
        root, history = mueller_root(function, x)
        if root is None:
            print(f"{str(x):>16} found no root")
        else:
            print(f"{str(x):>16} found root {root:.6f}")

    plot_x = numpy.linspace(-3.5, -0.5, 1000)
    pyplot.plot(plot_x, [function(x) for x in plot_x])
    pyplot.plot([-3.5, -0.5], [0, 0], "k--")
    pyplot.title("Double-check plot of 16x^4 + 144x^3 + 480x^2 + 692x + 359")
    pyplot.show()


def y_roots():
    roots = [
        -2.0840586945037827,
        -1.2979073167463429,
    ]
    for x in roots:
        p = lambda y: 2*x**2 + 2*y**2 + 8*x + 4*y + 9
        q = lambda y: x**2 + y**2 + 2*x*y + 5*x + 7*y + 8

        print("")
        for i in range(-5, 5):
            start = numpy.array([i-1, i, i+1])
            root, history = mueller_root(p, start)
            if root is None:
                print(f"{x:.3f}:{str(start):>12} found no root")
            else:
                root_str = f"{root:.6f}"
                print(f"{x:.3f}:{str(start):>12} found root {root_str:>20} for p"
                      f"\t&q_{{error}} = {q(root)}, &p_{{error}} = {p(root)}")


if __name__ == "__main__":
    main()
