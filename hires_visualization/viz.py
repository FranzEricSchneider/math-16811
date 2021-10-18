import itertools
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy


# Always
numpy.set_printoptions(suppress=True, precision=2)

# BE VERY CAREFUL ABOUT THE SIZE
MAX_RANGE = 30

# TODO: Make a class to name an identify functions
# TODO: Give different range limits per function
# TODO: Try to figure out how to link axes together
FUNCTIONS = [
    # function,      name,  *+ rand a  *+rand b   *+rand c
    (numpy.cos,      "cos", (20, -10), (1, -0.5), (1, -0.5)),
    (numpy.sin,      "sin", (20, -10), (1, -0.5), (1, -0.5)),
    (numpy.exp,      "exp", (1, -0.5), (1, -0.5), (1, -0.5)),
    (numpy.abs,      "abs", (1, -0.5), (1, -0.5), (1, -0.5)),
    (lambda x: x**2, "sqr", (1, -0.5), (1, -0.5), (1, -0.5)),
]


class RandomFunction:
    def __init__(self, function, name, a, b, c):
        self.function = function
        self.name = name
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, values):
        return self.a * self.function(self.b * values) + self.c

    def __repr__(self):
        return f"{self.a:.2f} * {self.name}({self.b:.2f} * x) + {self.c:.2f}"


def create_random_data(key, dim=5):
    RNG = numpy.random.default_rng(key)

    # Make positive ranges and then mirror to negative
    ranges = RNG.integers(low=MAX_RANGE/6, high=MAX_RANGE, size=dim)
    ranges = numpy.array([-ranges, ranges]).T

    # Make list of (unequally sized) linspaces along the axes
    # BE VERY CAREFUL ABOUT THE SIZE
    linspaces = [
        numpy.linspace(minmax[0], minmax[1], 2 * minmax[1])
        for minmax in ranges
    ]

    # Generate the data
    values = None
    functions = []
    for i, axis in enumerate(numpy.meshgrid(*linspaces, indexing="ij")):
        function, name, alim, blim, clim = RNG.choice(FUNCTIONS)
        function_obj = RandomFunction(
            function=function,
            name=name,
            a=alim[0] * RNG.random() + alim[1],
            b=blim[0] * RNG.random() + blim[1],
            c=clim[0] * RNG.random() + clim[1],
        )
        print(f"Function {i}: {function_obj}")

        if values is None:
            values = function_obj(axis)
        else:
            values += function_obj(axis)

        functions.append(function_obj)

    # THIS WILL BE VERY LARGE, AND ANY GENERATION CHANGE CAN MAKE IT GROW A LOT
    return linspaces, functions, values


def random_1D_plots(axes, functions, data, num_per_dim, key):
    RNG = numpy.random.default_rng(key)

    for i, (axis, function) in enumerate(zip(axes, functions)):
        for j in range(num_per_dim):
            # Generate a random point along each axis, then
            # override the one we care about with :
            indices = [RNG.choice(range(len(reaxis)))
                       for reaxis in axes]
            indices[i] = slice(None)
            indices = tuple(indices)

            # Plot and save
            figure, plotaxis = pyplot.subplots()
            plotaxis.plot(axis, data[indices], "bo")
            plotaxis.set_title(f"Axis {i}: {function}\nslice values {indices}")
            pyplot.savefig(f"axis{i}_try{j+1}.png")
            pyplot.close(figure)


def random_2D_plots(axes, functions, data, num, key):
    RNG = numpy.random.default_rng(key)

    # Choose between random combos
    possibilities = [i for i in itertools.product(range(len(axes)), repeat=2)
                     if i[0] != i[1]]
    for idx_a, idx_b in RNG.choice(possibilities, size=num, replace=False):

        indices = [0] * len(axes)
        indices[idx_a] = slice(None)
        indices[idx_b] = slice(None)
        indices = tuple(indices)

        A, B = numpy.meshgrid(axes[idx_a], axes[idx_b], indexing="ij")

        figure = pyplot.figure()
        axis = pyplot.axes(projection='3d')
        axis.scatter(A.flatten(), B.flatten(), data[indices].flatten(),
                     c=data[indices].flatten(), cmap="jet")
        axis.set_xlabel(f"Axis {idx_a}, {functions[idx_a]}")
        axis.set_ylabel(f"Axis {idx_b}, {functions[idx_b]}")
        axis.set_title(f"Axes {idx_a} vs {idx_b}, 0 on other axes")
        pyplot.savefig(f"3d_axis_{idx_a}_vs_{idx_b}.png")
        pyplot.close(figure)

        figure, axis = pyplot.subplots()
        if A.shape[0] == data[indices].shape[0]:
            axis.contour(A, B, data[indices], levels=20)
        else:
            axis.contour(A, B, data[indices].T, levels=20)
        axis.set_xlabel(f"Axis {idx_a}, {functions[idx_a]}")
        axis.set_ylabel(f"Axis {idx_b}, {functions[idx_b]}")
        axis.set_title(f"Axes {idx_a} vs {idx_b}, 0 on other axes")
        pyplot.savefig(f"3d_axis_{idx_a}_vs_{idx_b}_contour.png")
        pyplot.close(figure)


def random_vector(RNG, axes):
    # Random scales from -1 to 1
    scales = (RNG.random(size=len(axes)) * 2) - 1
    return numpy.array([scale * axis[-1]
                        for scale, axis in zip(scales, axes)])


def line_visualization(axes, functions, data, num, key):
    RNG = numpy.random.default_rng(key)

    for i in range(num):

        # Generate two random vectors
        vectors = [random_vector(RNG, axes) for _ in range(2)]

        # Scale between them
        x = numpy.linspace(0, 1, 100)
        y = []
        for alpha in x:
            combo = vectors[0] * alpha + vectors[1] * (1 - alpha)
            y.append(numpy.sum([
                function(value) for function, value in zip(functions, combo)
            ]))

        # Plot and save
        figure, plotaxis = pyplot.subplots()
        plotaxis.plot(x, y, "bo")
        plotaxis.set_title(f"Slice from {vectors[0]} to\n{vectors[1]}")
        plotaxis.set_xlabel("Sweep from one vector to the next")
        pyplot.savefig(f"random_1d_try{i+1}.png")
        pyplot.close(figure)


def center_ray_visualization(axes, functions, data, num, key):
    RNG = numpy.random.default_rng(key)

    center = numpy.zeros(len(functions))
    for i in range(num):

        # Generate two random axis vectors
        vectors = [random_vector(RNG, axes) for _ in range(2)]

        X, Y = numpy.meshgrid(numpy.linspace(-1, 1, 200),
                              numpy.linspace(-1, 1, 200),
                              indexing="ij")
        XY_vector = (numpy.outer(vectors[0], X.flatten()) +
                     numpy.outer(vectors[1], Y.flatten()))
        Z = numpy.sum([
            function(XY_vector[i])
            for i, function in enumerate(functions)
        ], axis=0)

        figure = pyplot.figure()
        axis = pyplot.axes(projection='3d')
        axis.scatter(X.flatten(), Y.flatten(), Z, c=Z, cmap="jet")
        axis.set_xlabel(f"Axis {vectors[0]}")
        axis.set_ylabel(f"Axis {vectors[1]}")
        axis.set_title(f"Random 2D viz, center 0")
        pyplot.savefig(f"random_2d_try{i+1}.png")
        pyplot.close(figure)

        figure, axis = pyplot.subplots()
        axis.contour(X, Y, Z.reshape(X.shape), levels=30)
        axis.set_xlabel(f"Axis {vectors[0]}")
        axis.set_ylabel(f"Axis {vectors[1]}")
        axis.set_title(f"Random 2D viz, center 0")
        pyplot.savefig(f"random_2d_try{i+1}_contour.png")
        pyplot.close(figure)


def main():
    # Create data and get some initial looks
    axes, functions, data = create_random_data(11111)
    # random_1D_plots(axes, functions, data, num_per_dim=2, key=22222)
    # random_2D_plots(axes, functions, data, num=10, key=33333)

    # Viz method #1
    # line_visualization(axes, functions, data, num=10, key=44444)

    # Viz method #2
    center_ray_visualization(axes, functions, data, num=10, key=55555)


if __name__ == "__main__":
    main()
