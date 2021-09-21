import argparse

from itertools import combinations, cycle, islice
from matplotlib import pyplot
import numpy
from scipy.spatial import Delaunay


# Location of the hoop
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
        return (self.x, self.y, "bo-" if self.is_cw else "go-")

    @property
    def plot_kwargs(self):
        return {"markersize": 2, "linewidth": 0.5}


class Triangle:
    def __init__(self, paths):
        assert len(paths) == 3

        # Enforce a CW ordering of the three points, so that they orbit the
        # center in a CW manner. If the cross is negative (Z into the page)
        # that means we're rotating CW
        cross = numpy.cross(vector(paths[0], paths[1]),
                            center(paths) - paths[0].start)
        if cross <= 0:
            self.paths = paths
        else:
            self.paths = [paths[0], paths[2], paths[1]]

    def contains(self, point):
        points = [p.start for p in self.paths]
        for p0, p1 in zip(points, islice(cycle(points), 1, None)):
            # Get a vector going into the triangle, then see if the center - p0
            # has a positive or negative dot product with that vector
            perpendicular = RN90.dot(p1 - p0)
            unit = perpendicular / numpy.linalg.norm(perpendicular)
            if unit.dot(self.center - p0):
                return False
        return True

    def score(self, point):
        pass

    @property
    def center(self):
        try:
            return self._center
        except AttributeError:
            self._center = center(self.paths)
            return self._center

    @property
    def plot_args(self):
        return [
            (path.x[0], path.y[0], color + "o")
            for path, color in zip(self.paths, "rgb")
        ]


def center(paths):
    """Return the center of the start points of the given paths."""
    return numpy.average(
        numpy.array([[p.x[0], p.y[0]] for p in paths]),
        axis=0,
    )


def vector(p0, p1):
    """Return a vector from the start of p0 to the start of p1."""
    return p1.start - p0.start


def plot_scene(paths=tuple(), fire=True, dest=True, delaunay=None):
    figure = pyplot.figure()
    axis = figure.add_subplot(111)

    for path in paths:
        axis.plot(*path.plot_args, **path.plot_kwargs)
    if fire:
        circle = pyplot.Circle(
            tuple(FIRE_PT), FIRE_RADIUS, color='r', fill=False, linewidth=2
        )
        axis.add_artist(circle)
    if dest:
        axis.plot([8], [8], "ro", markersize=15)
    if delaunay:
        for simplex in delaunay.simplices:
            for idx0, idx1 in zip(simplex, islice(cycle(simplex), 1, None)):
                p0 = delaunay.points[idx0]
                p1 = delaunay.points[idx1]
                axis.plot([p0[0], p1[0]], [p0[1], p1[1]], "k", linewidth=1)

    axis.set_xlim(0, 12)
    axis.set_ylim(0, 12)
    axis.set_aspect('equal', adjustable='box')
    pyplot.show()


def main(start, paths, plot_contains, plot_delaunay):
    cw = [path for path in paths if path.is_cw]
    cw_points = numpy.array([path.start for path in cw])
    ccw = [path for path in paths if not path.is_cw]
    ccw_points = numpy.array([path.start for path in ccw])

    if plot_delaunay:
        cw_delaunay = Delaunay(cw_points)
        plot_scene(paths, delaunay=cw_delaunay)
        ccw_delaunay = Delaunay(ccw_points)
        plot_scene(paths, delaunay=ccw_delaunay)
    import ipdb; ipdb.set_trace()

    triangles = []
    for tri in combinations(cw, r=3):
        triangles.append(Triangle(tri))
    for tri in combinations(ccw, r=3):
        triangles.append(Triangle(tri))

    # Optional debugging of the "contain" process
    if plot_contains:
        for tri in triangles:
            for point_args in tri.plot_args:
                pyplot.plot(*point_args, markersize=15)
            if tri.contains(start):
                color = "r"
            else:
                color = "g"
            pyplot.plot(start[0], start[1], color + "o", markersize=20)
            pyplot.show()
    import ipdb; ipdb.set_trace()
    pass


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

    for starting_point in STARTING_POINTS:
        main(starting_point,
             paths,
             plot_contains=args.plot_contains,
             plot_delaunay=args.plot_delaunay,
             )
