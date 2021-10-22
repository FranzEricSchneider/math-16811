import argparse
import numpy
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def main(plot_initial):

    obstacle_cost = generate_cost()

    start_point = numpy.array([10, 10])
    end_point = numpy.array([90, 90])
    vector = end_point - start_point

    num_pts = 100
    initial_path = start_point + \
        numpy.outer(numpy.linspace(0, 1, num_pts), vector)

    if plot_initial:
        plot_scene(initial_path, obstacle_cost)

    naive_1step_path, _ = naive_gradient(initial_path, obstacle_cost, steps=1)
    plot_scene(naive_1step_path, obstacle_cost)
    naive_path, _ = naive_gradient(initial_path, obstacle_cost)
    plot_scene(naive_path, obstacle_cost)


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
    # If this assertion fails, rethink the x, y step
    assert space2D.shape[0] == space2D.shape[1]
    x, y = numpy.clip(path.astype(int).T, 0, space2D.shape[0] - 1)
    return space2D[x, y]


def plot_scene(path, cost):
    values = get_values(path, cost)

    # Plot 2D
    pyplot.imshow(cost.T)
    pyplot.plot(path[:, 0], path[:, 1], "ro", ms=1)

    # Plot 3D
    fig3d = pyplot.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    xx, yy = numpy.meshgrid(range(cost.shape[1]), range(cost.shape[0]))
    ax3d.plot_surface(xx, yy, cost.T, cmap=pyplot.get_cmap("coolwarm"))
    ax3d.scatter(path[:, 0], path[:, 1], values, s=20, c="r")
    pyplot.show()


def naive_gradient(path, cost, rate=0.1, steps=0):
    # Re-check clip step if this fails
    assert cost.shape[0] == cost.shape[1]
    # Don't modify the original path
    path = path.copy()
    # Get the global gradients and evaluate our position on it over time
    gx, gy = numpy.gradient(cost)
    # Keep looping until values stop changing very much
    count = 0
    sqr_step = 1000
    while sqr_step > 1e-5:
        # Step downwards
        last_path = path.copy()
        step = -rate * numpy.vstack([get_values(path, gx),
                                     get_values(path, gy)]).T
        path += step
        path = numpy.clip(path, 0, cost.shape[0])
        # See how big the step was to evaluate ending
        sqr_step = numpy.sum((last_path - path)**2)
        # Bookkeeping at the end
        count += 1
        print(f"count: {count}")
        # Check if there is an early ending criteria
        if steps > 0:
            if count == steps:
                print("count == steps!!")
                return path, count

    return path, count



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path through obstacles")
    parser.add_argument("-i", "--plot-initial",
                        help="Plot the initial setup.",
                        action="store_true")
    args = parser.parse_args()
    main(args.plot_initial)
