import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def main():
    critical_points = numpy.array([[0, 0], [-2, 0], [0, 4/3], [-2, 4/3]])

    x, y = numpy.meshgrid(numpy.linspace(-2.5, 0.5, 100),
                          numpy.linspace(-0.5, 0.5 + (4/3), 100))
    plot_contours(x, y, function(x, y), critical_points,
                  title="Contour of function, critical points")

    x, y = numpy.meshgrid(
        numpy.linspace(-2.5, 0.5, 31),  # Hits -2.0 and 0.0
        numpy.linspace(-1/3, 1/3 + (4/3), 31)  # Hits 0.0 and 4/3
    )
    plot_gradients(x, y, critical_points)

    x, y = numpy.meshgrid(numpy.linspace(-4, 2, 50),
                          numpy.linspace(-2, 2 + (4/3), 50))
    plot_3d(x, y, function(x, y), critical_points,
            title="3d of function, critical points")

    pyplot.show()


def function(x, y): return x**3 + y**3 + 3*x**2 - 2*y**2 - 8
def ddx(x): return 3 * x**2 + 6 * x
def ddy(y): return 3 * y**2 - 4 * y


def plot_contours(x, y, z, points, title=None):
    fig2d, ax2d = pyplot.subplots()
    contours = ax2d.contour(x, y, z, levels=40)
    pyplot.clabel(contours, inline=1, fontsize=10, levels=contours.levels[::2])
    ax2d.plot(points[:, 0], points[:, 1], "go", ms=10)
    ax2d.set_aspect("equal", "box")
    if title is not None:
        pyplot.title(title)

def plot_3d(x, y, z, points, title=None):
    fig3d = pyplot.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.scatter(points[:, 0],
                 points[:, 1],
                 function(points[:, 0], points[:, 1]),
                 s=75,
                 c="g")
    ax3d.plot_surface(x, y, z.T, cmap=pyplot.get_cmap("coolwarm"), alpha=0.75)
    if title is not None:
        ax3d.set_title(title)


def plot_gradients(x, y, points):
    fig2d, axes = pyplot.subplots(2, 2)
    for axis in (axes[0, 0], axes[1, 0], axes[0, 1]):
        axis.plot(points[:, 0], points[:, 1], "go", ms=10)
        axis.set_aspect("equal", "box")

    dx = ddx(x)
    dy = ddy(y)
    norm = numpy.linalg.norm(
        numpy.vstack((dx.flatten(), dy.flatten())),
        axis=0,
    )
    axes[0, 0].quiver(x, y, dx, dy*0, norm)
    axes[0, 0].set_title("Visualizing X gradient only")
    axes[1, 0].quiver(x, y, dx*0, dy, norm)
    axes[1, 0].set_title("Visualizing Y gradient only")
    axes[0, 1].set_title("(X, Y) gradient")
    axes[0, 1].quiver(x, y, dx, dy, norm)


if __name__ == "__main__":
    main()
