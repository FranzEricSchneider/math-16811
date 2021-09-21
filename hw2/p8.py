import argparse

from itertools import combinations, cycle, islice
from matplotlib import pyplot
import numpy
from scipy.spatial import Delaunay

from p1 import poly_interp


# Location and size of the hoop
FIRE_PT = numpy.array([5, 5])
FIRE_RADIUS = 1.5

STARTING_POINTS = [
    numpy.array([0.8, 1.8]),
    numpy.array([2.2, 1.0]),
    numpy.array([2.7, 1.4]),
]

# Rotation matrix for -90 degrees
RN90 = numpy.array([[ 0, 1],
                    [-1, 0]])


class Path:
    def __init__(self, line0, line1):
        self.x = numpy.array([float(element) for element in line0.split()])
        self.y = numpy.array([float(element) for element in line1.split()])

    @property
    def start(self):
        return numpy.array([self.x[0], self.y[0]])

    @property
    def is_cw(self):
        """
        Return boolean for whether the path goes clockwise (cw/left/over)
        around the hoop. A False value means it went counter-clockwise
        (ccw/right/under).
        """
        try:
            return self._is_cw
        except AttributeError:
            # Get a sub-sampled set of points along the path, then take their
            # difference from the fire for each point.
            samples = numpy.vstack((self.x[::10], self.y[::10])).T - FIRE_PT
            crosses = []
            for p0, p1 in zip(samples[:-1], samples[1:]):
                # Use the cross product to see if we're rotating CW (z axis
                # into the page) or CCW (z axis out of the page)
                crosses.append(numpy.cross(p0, p1))
            # The given data is simple - the signs of all crosses should match.
            # Re-evaluate if it ever happens otherwise
            assert all([numpy.sign(crosses[0]) == numpy.sign(next_angle)
                        for next_angle in crosses[1:]])
            self._is_cw = numpy.sign(crosses[0]) < 0
            return self._is_cw

    @property
    def plot_args(self):
        return (self.x, self.y, "bo-" if self.is_cw else "co-")

    @property
    def plot_kwargs(self):
        return {"markersize": 2, "linewidth": 0.5}


class Triangle:
    def __init__(self, paths):
        assert len(paths) == 3

        # Enforce a CW ordering of the three points, so that they orbit the
        # center in a CW manner. If the cross is negative (Z into the page)
        # that means we're rotating CW
        cross = numpy.cross(paths[1].start - paths[0].start,
                            center(paths) - paths[0].start)
        if cross <= 0:
            self.paths = paths
        else:
            self.paths = [paths[0], paths[2], paths[1]]

    def contains(self, point):
        for p0, p1 in self.cycle:
            # Get a vector going into the triangle, then see if the center - p0
            # has a positive or negative dot product with that vector
            perpendicular = RN90.dot(p1 - p0)
            unit = perpendicular / numpy.linalg.norm(perpendicular)
            if unit.dot(point - p0) < 0:
                return False
        return True

    def svd_weights(self, point):
        """
        In order to find the weights to come to a point somewhere within a
        given triangle, we can use SVD. We have infinite exact solutions, b/c
        we have three (assumedly independent) vectors covering a 2D space.
        Therefore the null space is rank 1. We can get an SVD solution on how
        to reach a particular point with those three vectors, which will be one
        possible set of weights.
        """
        A = self.points.T
        u, s, vt = numpy.linalg.svd(A)
        # Make sigma_inv a matrix
        sigma_inv = numpy.zeros(A.T.shape)
        for i, s_value in enumerate(s):
            sigma_inv[i, i] = 1 / s_value
        # Find the SVD solution for this given point in space
        weights = vt.T @ sigma_inv @ u.T @ point

        # TODO: Figure out a final weight solution. This solution to get
        # weights with a unit norm was a failure, the path diverged extensively
        # # null space = unit norm
        # null = vt[2, :]
        # scalar = numpy.sqrt(1 - numpy.linalg.norm(weights)**2)
        # assert numpy.isclose(weights.dot(null), 0.0)
        # return weights + scalar * null

        # TODO: Figure out a final weight solution. This solution using just
        # the average tracks the paths nicely but doesn't hit the start exactly
        # return numpy.array([1, 1, 1]) / 3

        return weights

    @property
    def points(self):
        return numpy.array([p.start for p in self.paths])

    @property
    def cycle(self):
        return zip(self.points, islice(cycle(self.points), 1, None))

    @property
    def center(self):
        try:
            return self._center
        except AttributeError:
            self._center = center(self.paths)
            return self._center

    @property
    def plot_point_args(self):
        return [
            (path.x[0], path.y[0], color + "o")
            for path, color in zip(self.paths, "rgb")
        ]

    @property
    def plot_line_args(self):
        return [
            ([p0[0], p1[0]], [p0[1], p1[1]])
            for p0, p1 in self.cycle
        ]


class Interpolation:
    def __init__(self, paths, weights):
        self.paths = paths
        self.weights = weights
        # See the poly_interp function for what this does. We want one cache
        # for x and another for y, for each path
        self.derivative_cache = [[{}, {}] for path in paths]

    def plot(self, axis):
        # Plot the paths made up by the original triangle
        axis.plot([-1], [-1], "ko", ms=4, lw=1, label="triangle paths")
        for path in self.paths:
            axis.plot(path.x, path.y, "ko-", ms=4, lw=1)

        time = numpy.linspace(0, len(self.paths[0].x) - 1, 100)
        interp_xy = numpy.array([self.interpolate(t) for t in time])
        axis.plot([-1], [-1], "g", lw=3, label="interpolated")
        axis.plot(interp_xy[:, 0], interp_xy[:, 1], "g", lw=3)

    def interpolate(self, time):
        """Interpolated weighted paths at the given time.

        It is assumed that we step through the original paths at one step per
        time step, so time=20 corresponds to path.x[20]
        """

        # Find a 4-element window within which to interpolate. I chose 4
        # elements because then the interpolated data points (X) would be
        # symmetric around the time point (o) and I wouldn't have to figure out
        # something clever for odd-numbered windows
        # X....X..o.X....X
        radius = 2
        # Clamp the lower radius to 0 (makes sense) and also so that lower
        # plus 2*radius hits the end of the path.
        lower_idx = numpy.clip(a=int(time) - radius,
                               a_min=0,
                               a_max=len(self.paths[0].x) - (2 * radius))
        upper_idx = lower_idx + 2 * radius

        # IMPORTANT - We are interpolating SEPARATELY on both x and y
        # Create a 2x3 matrix of [x1, x2, x3]
        #                        [y1, y2, y3]
        # Where the (x, y) values are the interpolated values on each path
        time_data = list(range(lower_idx, upper_idx))
        points = numpy.array([
            [
                poly_interp(eval_at=time,
                            X=time_data,
                            Y=path.x[lower_idx:upper_idx],
                            deriv_cache=self.derivative_cache[i][0]),
                poly_interp(eval_at=time,
                            X=time_data,
                            Y=path.y[lower_idx:upper_idx],
                            deriv_cache=self.derivative_cache[i][1]),
            ]
            for i, path in enumerate(self.paths)
        ]).T
        xy = points.dot(self.weights)

        # Check our interpolation at each data point
        if numpy.isclose(time, int(time)):
            true_points = numpy.array([
                [self.paths[i].x[int(time)] for i in range(len(self.paths))],
                [self.paths[i].y[int(time)] for i in range(len(self.paths))],
            ])
            assert numpy.allclose(xy, true_points.dot(self.weights))

        return xy


def center(paths):
    """Return the center of the start points of the given paths."""
    return numpy.average(
        numpy.array([[p.x[0], p.y[0]] for p in paths]),
        axis=0,
    )


def plot_scene(paths=tuple(), fire=True, dest=True, delaunay=None,
               tripoint=None, interpath=None):
    figure = pyplot.figure()
    axis = figure.add_subplot(111)

    for path in paths:
        axis.plot(*path.plot_args, **path.plot_kwargs)
    if fire:
        circle = pyplot.Circle(
            tuple(FIRE_PT), FIRE_RADIUS, color='r', fill=False, lw=2
        )
        axis.add_artist(circle)
    if dest:
        axis.plot([8], [8], "ro", ms=15)
    if delaunay:
        for simplex in delaunay.simplices:
            for idx0, idx1 in zip(simplex, islice(cycle(simplex), 1, None)):
                p0 = delaunay.points[idx0]
                p1 = delaunay.points[idx1]
                axis.plot([p0[0], p1[0]], [p0[1], p1[1]], "k", lw=1)
    if tripoint:
        point, triangles = tripoint
        axis.plot(point[0], point[1], "ko", ms=10)
        for tri in triangles:
            color = "r"
            width = 0.5
            if tri.contains(point):
                color = "g"
                width = 3
            for line_args in tri.plot_line_args:
                pyplot.plot(*line_args, color, lw=width)
    if interpath:
        interpath.plot(axis)

    axis.set_xlim(0, 12)
    axis.set_ylim(0, 12)
    axis.set_aspect('equal', adjustable='box')
    pyplot.show()


def main(paths, plot_contains, plot_delaunay, plot_weights, plot_paths):
    cw = [path for path in paths if path.is_cw]
    ccw = [path for path in paths if not path.is_cw]
    ccw_points = numpy.array([path.start for path in ccw])

    triangles = []
    for path_grouping in [cw, ccw]:
        points = numpy.array([path.start for path in path_grouping])
        delaunay = Delaunay(points)
        for simplex in delaunay.simplices:
            triangles.append(Triangle([path_grouping[idx] for idx in simplex]))
        if plot_delaunay:
            plot_scene(paths, delaunay=delaunay)

    for start in STARTING_POINTS:

        # Optional debugging of the "contain" process
        if plot_contains:
            plot_scene(paths, tripoint=(start, triangles))

        # By the HW assertions we should have one single triangle here
        chosen = [tri for tri in triangles if tri.contains(start)][0]

        # Find the SVD solution for this given point in space
        # PROBLEM - BASIC SVD SOLUTION DOES NOT HAVE UNIT NORM
        weights = chosen.svd_weights(start)
        # TODO: Try getting barycentric weights?

        # TODO: Refactor this into a function?
        if plot_weights:
            for point_args in chosen.plot_point_args:
                pyplot.plot(*point_args)
            p0 = numpy.array([0, 0])
            for i, (tripoint, color) in enumerate(zip(chosen.points, "rgb")):
                p1 = p0 + weights[i] * chosen.points[i]
                pyplot.plot([0, tripoint[0]], [0, tripoint[1]], color)
                pyplot.plot([p0[0], p1[0]], [p0[1], p1[1]], color, lw=3)
                p0 = p1
            pyplot.plot(start[0], start[1], "ko", ms=15)
            pyplot.show()

        # Calculate the interpolated path
        interpolation = Interpolation(chosen.paths, weights)
        # But only display it if requested
        if plot_paths:
            plot_scene(paths, interpath=interpolation)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Unicycle pather.")
    parser.add_argument("-c", "--plot-contains",
                        help="Plot debug output for the contains step.",
                        action="store_true")
    parser.add_argument("-d", "--plot-delaunay",
                        help="Plot debug output for the delaunay step.",
                        action="store_true")
    parser.add_argument("-i", "--initial-state",
                        help="Plot the initial input state.",
                        action="store_true")
    parser.add_argument("-p", "--plot-paths",
                        help="Plot debug output for the final paths.",
                        action="store_true")
    parser.add_argument("-w", "--plot-weights",
                        help="Plot debug output for the weight solution.",
                        action="store_true")
    args = parser.parse_args()

    paths = []
    with open("p8_paths.txt", "r") as file:
        while True:
            line0 = file.readline()
            line1 = file.readline()
            if line0:
                paths.append(Path(line0, line1))
            else:
                break
    if args.initial_state:
        plot_scene(paths)

    main(paths,
         plot_contains=args.plot_contains,
         plot_delaunay=args.plot_delaunay,
         plot_paths=args.plot_paths,
         plot_weights=args.plot_weights,
         )
