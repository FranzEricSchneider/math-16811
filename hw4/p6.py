import argparse
import numpy
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def main(plot_initial):

    obstacle_cost = generate_cost()

    start_point = numpy.array([10, 10])
    end_point = numpy.array([90, 90])
    endpoints = numpy.vstack((start_point, end_point))
    vector = end_point - start_point

    num_pts = 100
    initial_path = start_point + \
        numpy.outer(numpy.linspace(0, 1, num_pts), vector)

    if plot_initial:
        plot_scene(initial_path, obstacle_cost, endpoints)

    # # Part A
    # naive_1step, _ = iterative_path(initial_path,
    #                                 obstacle_cost,
    #                                 TODOTHING,
    #                                 rate=8,
    #                                 steps=1)
    # plot_scene(naive_1step, obstacle_cost, endpoints,
    #            title="A) Naive Gradient 1 Step (rate 8x)")
    # naive, _ = iterative_path(initial_path, obstacle_cost, TODOTHING)
    # plot_scene(naive, obstacle_cost, endpoints,
    #            title="A) Naive Gradient Converged")

    # Part B
    for steps in (100, 200, 500):
        dist1d, _ = iterative_path(initial_path,
                                   obstacle_cost,
                                   OTHERTHING,
                                   steps=steps)
        plot_scene(dist1d, obstacle_cost, endpoints,
                   title=f"B) 1D Smoothing {steps} steps")


def generate_cost():
    n = 101
    obstacles = numpy.array([[20, 30], [60, 40], [70, 85]])
    epsilon = numpy.array([[25], [20], [30]])
    obstacle_cost = numpy.zeros((n, n))
    for i in range(obstacles.shape[0]):
        t = numpy.ones((n, n))
        t[obstacles[i, 0], obstacles[i, 1]] = 0
        t_cost = distance_transform_edt(t)
        t_cost[t_cost > epsilon[i]] = epsilon[i]
        t_cost = (1 / (2 * epsilon[i])) * (t_cost - epsilon[i])**2
        obstacle_cost += + t_cost
    return obstacle_cost


def get_values(path, space2D):
    """
    Path is a (N, 2) vector
    space2D is an (M, M) matrix
    """
    # If this assertion fails, rethink the x, y step
    assert space2D.shape[0] == space2D.shape[1]
    x, y = numpy.clip(path.astype(int).T, 0, space2D.shape[0] - 1)
    return space2D[x, y]


def plot_scene(path, cost, endpoints, title=None):

    # Plot 2D
    pyplot.imshow(cost.T)
    pyplot.plot(endpoints[:, 0], endpoints[:, 1], "go", ms=10)
    if title is not None:
        pyplot.title(title)
    pyplot.plot(path[:, 0], path[:, 1], "ro", ms=1)

    # Plot 3D
    fig3d = pyplot.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    xx, yy = numpy.meshgrid(range(cost.shape[1]), range(cost.shape[0]))
    ax3d.plot_surface(xx, yy, cost.T, cmap=pyplot.get_cmap("coolwarm"), alpha=0.6)
    ax3d.scatter(endpoints[:, 0],
                 endpoints[:, 1],
                 get_values(endpoints, cost),
                 s=50,
                 c="g")
    ax3d.scatter(path[:, 0],
                 path[:, 1],
                 get_values(path, cost),
                 s=20,
                 c="r")
    if title is not None:
        ax3d.set_title(title)
    pyplot.show()


def iterative_path(path, cost, THING, rate=0.1, steps=0, step_thresh=1e-5):
    # Re-check clip step if this fails
    assert cost.shape[0] == cost.shape[1]
    # Don't modify the original path
    path = path.copy()
    # Get the global gradients and evaluate our position on it over time
    gx, gy = numpy.gradient(cost)
    # Keep looping until values stop changing very much
    count = 0
    sqr_step = 1000
    while sqr_step > step_thresh:
        # Step downwards
        last_path = path.copy()
        step = -rate * THING(path, gx, gy)
        # Don't modify the endpoints
        path[1:-1] += step[1:-1]
        # Enforce that we don't go outside the area
        path = numpy.clip(path, 0, cost.shape[0] - 1)
        # See how big the step was to evaluate ending
        sqr_step = numpy.sum((last_path - path)**2)
        # Bookkeeping at the end
        count += 1
        # Check if there is an early ending criteria
        if steps > 0:
            if count == steps:
                return path, count

    return path, count


def TODOTHING(path, gx, gy):
    return numpy.vstack([get_values(path, gx),
                         get_values(path, gy)]).T


def OTHERTHING(path, gx, gy):
    """
    Returns shape (N, 2)
    """

    gradient_weight = 0.8
    smooth_weight = 4.0

    gradient_force = gradient_weight * \
                     numpy.vstack([get_values(path, gx), get_values(path, gy)])

    norm = numpy.linalg.norm(path[1:] - path[:-1], axis=1)
    direction = (path[1:] - path[:-1]).T / norm
    smooth_force = numpy.zeros(gradient_force.shape)
    smooth_force[:, 1:] = smooth_weight * (0.5 * norm**2) * direction

    return (gradient_force + smooth_force).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path through obstacles")
    parser.add_argument("-i", "--plot-initial",
                        help="Plot the initial setup.",
                        action="store_true")
    args = parser.parse_args()
    main(args.plot_initial)
