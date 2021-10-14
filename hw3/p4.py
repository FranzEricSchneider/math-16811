from matplotlib import pyplot
from mpl_toolkits import mplot3d
import numpy

RNG = numpy.random.default_rng(12345)


# Always
numpy.set_printoptions(suppress=True, precision=3)


def read_values(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    return numpy.array([[float(x) for x in line.split()]
                        for line in lines])


def plane_distance(xyz, weights, average=True):
    """
    See section 4.1.2 for the equation.
    xyz should be (N, 3)
    weights should be (4,)
    """

    # This assertion must be true for the plane distance to work
    assert numpy.isclose(numpy.linalg.norm(weights[:3]), 1.0)

    projection = xyz.dot(weights[:3])
    distance = numpy.abs(projection + weights[3])
    if average:
        return numpy.average(distance)
    else:
        return distance


def fit_plane(xyz):
    u, s, vt = numpy.linalg.svd(xyz)
    sigma_inv = numpy.zeros(xyz.T.shape)
    for i, s_value in enumerate(s):
        sigma_inv[i, i] = 1 / s_value

    # TODO: Explain
    d_vector = -1 * numpy.ones(xyz.shape[0])

    # Calculate weights, then add in the forced d=1 setting
    weights = vt.T @ sigma_inv @ u.T @ d_vector
    weights = numpy.hstack((weights, [1]))
    # Normalize around (a, b, c) (the direction vector)
    weights /= numpy.linalg.norm(weights[:3])
    return weights


def p4a(filename):
    xyz_values = read_values(filename)
    plot_raw(xyz_values, filename)
    weights = fit_plane(xyz_values)
    print(f"Weights from {filename}: {weights}")
    print(f"Average distance from plane: {plane_distance(xyz_values, weights, average=True) * 1000}mm")
    print(f"{xyz_values.shape} points")
    plot_processed(xyz_values, filename, weights)


def plot_raw(xyz, filename):
    figure = pyplot.figure()
    axis = pyplot.axes(projection='3d')
    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "bo")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"Raw values from {filename}")
    pyplot.show()


def plane_z_from_xy(X, Y, weights):
    """
    We're working with the plane equation ax + by + cz + d = 0. Therefore
        -cz = ax + by + d
        z = -(ax + by + d) / c
    """
    a, b, c, d = weights
    return -(a*X + b*Y + d) / c


def plot_processed(xyz, filename, weights, n=50):
    figure = pyplot.figure()
    axis = pyplot.axes(projection='3d')

    inliers = xyz[plane_distance(xyz, weights, average=False) < 0.02]

    x = numpy.linspace(numpy.min(inliers[:, 0]), numpy.max(inliers[:, 0]), n)
    y = numpy.linspace(numpy.min(inliers[:, 1]), numpy.max(inliers[:, 1]), n)
    X, Y = numpy.meshgrid(x, y)
    Z = plane_z_from_xy(X, Y, weights)
    axis.plot(X.flatten(), Y.flatten(), Z.flatten(), "ro-", label="Fit")

    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "bo", label="Raw")

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"{filename} values compared to plane, (a, b, c, d) = {weights}")
    axis.legend()
    pyplot.show()


def plot_multi_processed(xyz, filename, weight_groups, n=50, zlim=None):
    figure = pyplot.figure()
    axis = pyplot.axes(projection='3d')

    for weights in weight_groups:
        inliers = xyz[plane_distance(xyz, weights, average=False) < 0.02]

        x = numpy.linspace(numpy.min(inliers[:, 0]), numpy.max(inliers[:, 0]), n)
        y = numpy.linspace(numpy.min(inliers[:, 1]), numpy.max(inliers[:, 1]), n)
        X, Y = numpy.meshgrid(x, y)
        Z = plane_z_from_xy(X, Y, weights).flatten()
        mask = numpy.ones(Z.shape[0], dtype=bool)
        if zlim:
            mask &= Z > zlim[0]
            mask &= Z < zlim[1]
        axis.plot(X.flatten()[mask], Y.flatten()[mask], Z[mask], "ro-")

        axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "bo")

    if zlim:
        axis.set_zlim(zlim)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"{filename} values compared to {len(weight_groups)} fit planes")
    pyplot.show()


def construct_matrix(xvals, order=5):
    """Construct a polynomal matrix of the x values like so:
        [1, x[0], x[0]^2, x[0]^3, ...]
        ...
        [1, x[n], x[n]^2, x[n]^3, ...]
    For the given order of polynomial, where n is the length of xvals.
    """
    return numpy.array([
        [x**i for i in range(order + 1)]
        for x in xvals
    ])


def ransac_plane(xyz, min_frac=0.25, batch=12, clean_std=2.07e-3):

    # Choose a threshold that is a certain number of standard deviations from
    # a clean fit to good data
    threshold = 4 * clean_std
    # Make a vector mask to check which variables have already been claimed
    unclaimed = numpy.ones(xyz.shape[0], dtype=bool)
    # Number of points needed
    min_num_fit = min_frac * len(unclaimed)
    # Models found (weights)
    models = []
    # Quality of said models
    errors = []

    while numpy.sum(unclaimed) > min_num_fit:

        # Make an arbitrary set of weights as a starter set
        best_model = numpy.array([1, 1, 1, 1])
        best_error = 1000
        # These are the indices available for choosing this round
        unclaimed_indices = numpy.argwhere(unclaimed).flatten()
        # Check each time whether we founc anything
        got_new_model = False

        for _ in range(batch):
            # Select three random points and calculate weights
            chosen_idx = RNG.choice(unclaimed_indices, size=3, replace=False)
            chosen = xyz[chosen_idx]
            weights = fit_plane(chosen)
            # plot_processed(chosen, None, weights)  # Uncomment to see chosen

            # Check if the plane 1) has enough points and 2) is a better model
            inlier_mask = plane_distance(xyz[unclaimed], weights, average=False) < threshold
            if numpy.sum(inlier_mask) > min_num_fit:
                better_weights = fit_plane(xyz[unclaimed][inlier_mask])
                better_distances = plane_distance(xyz[unclaimed], better_weights, average=False)
                better_inlier_mask = better_distances < threshold
                # TODO: Is std a good error metric? Would avg distance be better?
                # I feel like avg distance may sort of be covered already by the
                # threshold
                error = numpy.std(better_distances[better_inlier_mask])
                if error < best_error:
                    best_model = better_weights

        # This indicates that the last search came up empty, to just return
        # what we already have
        if numpy.allclose(best_model, 1.0):
            return models, errors, unclaimed

        # Do some bookkeeping to update unclaimed and save the model
        final_inlier_mask = plane_distance(xyz, best_model, average=False) < threshold
        unclaimed &= numpy.logical_not(final_inlier_mask)
        models.append(best_model)
        errors.append(best_error)

    return models, errors, unclaimed


def p4c(filename):

    xyz_values = read_values(filename)
    plot_raw(xyz_values, filename)

    weight_groups, _, unclaimed = ransac_plane(xyz_values, min_frac=0.6, batch=12)
    assert len(weight_groups) == 1
    weights = weight_groups[0]

    print(f"Weights after RANSAC from {filename}: {weights}")
    print(f"Percent of points used {100 * (1 - (numpy.sum(unclaimed) / unclaimed.shape[0]))}%")
    print("Average inlier distance from plane: "
          f"{plane_distance(xyz_values[numpy.logical_not(unclaimed)], weights, average=True) * 1000}mm")
    print(f"Average distance from plane: {plane_distance(xyz_values, weights, average=True) * 1000}mm")
    plot_processed(xyz_values, filename, weights)


def p4d(filename):

    xyz_values = read_values(filename)
    plot_raw(xyz_values, filename)

    weight_groups, _, unclaimed = ransac_plane(xyz_values, min_frac=0.10, batch=100)
    print(f"Found {len(weight_groups)} weight groups")
    print(f"{100 * (numpy.sum(unclaimed) / unclaimed.shape[0])}% of points unrepresented")
    print(f"Weights after RANSAC from {filename}: {weight_groups}")

    plot_multi_processed(xyz_values, filename, weight_groups, zlim=(0, 3))


def p4e(filename):

    xyz_values = read_values(filename)
    plot_raw(xyz_values, filename)

    v1_weight_groups, _, v1_unclaimed = ransac_plane(
        xyz_values, min_frac=0.2, batch=50, clean_std=0.01
    )
    print(f"Found {len(v1_weight_groups)} weight groups")

    # Uncomment to view the remaining points
    # plot_raw(xyz_values[v1_unclaimed], "V2_cluttered_hallway")

    v2_weight_groups, _, v2_unclaimed = ransac_plane(
        xyz_values[v1_unclaimed], min_frac=0.4, batch=150, clean_std=0.04
    )
    print(f"Found {len(v2_weight_groups)} more weight groups in the second pass")
    print(f"V1 weights after RANSAC from {filename}: {v1_weight_groups}")
    print(f"V2 weights after RANSAC from {filename}: {v2_weight_groups}")
    print(f"{100 * (numpy.sum(v2_unclaimed) / xyz_values.shape[0])}% of points unrepresented")

    weight_groups = v1_weight_groups + v2_weight_groups
    plot_multi_processed(xyz_values, filename, weight_groups, zlim=(1, 6))

    for weights, threshold, name in zip(weight_groups,
                                        [0.04, 0.04, 0.04, 0.04],
                                        ["Pass 1", "Pass 1", "Pass 2", "Pass 2"]):
        project_wall(xyz_values, weights, threshold, name)


def project_wall(xyz, weights, threshold, name):
    # Consider only the inliers
    inliers = xyz[plane_distance(xyz, weights, average=False) < threshold]
    distances = plane_distance(inliers, weights, average=False)

    # Do Gram-Schmidt to get orthogonal axes perpendicular to the plane
    def gram_schmidt(axis, references):
        for reference in references:
            axis -= reference * (axis.dot(reference) / numpy.linalg.norm(reference))
        return axis / numpy.linalg.norm(axis)
    axis_0 = gram_schmidt(axis=numpy.array([1, 0, 0], dtype=float),
                          references=[weights[:3]])
    axis_1 = gram_schmidt(axis=numpy.array([0, 1, 0], dtype=float),
                          references=[weights[:3], axis_0])
    x = inliers.dot(axis_0)
    y = inliers.dot(axis_1)

    # Do related print statements
    RMS = numpy.sqrt(numpy.average(distances**2))
    density = len(x) / ((x.max() - x.min()) * (y.max() - y.min()))
    print(f"{weights}: RMS is {RMS}m")
    print(f"{weights}: Density is {density} pts/m^2")

    # Colormapped scatter plot
    pyplot.scatter(x, y, c=1000*distances, cmap="jet")
    pyplot.gca().set_aspect('equal', 'box')
    colorbar = pyplot.colorbar()
    colorbar.set_label('Distance from plane (mm)')
    pyplot.title(f"{name}: {weights}\nRMS: {RMS:.3f}m, density: {density:.1f} pts/m^2")
    pyplot.xlabel("Axis 0 (x with plane elements removed) (m)")
    pyplot.ylabel("Axis 1 (y with Axis 0 and plane elements removed) (m)")
    pyplot.show()


if __name__ == "__main__":
    # Part A
    print("\nPart A")
    p4a("clear_table.txt")

    # Part B can just use the same code
    print("\nPart B")
    p4a("cluttered_table.txt")

    # Part C
    print("\nPart C")
    p4c("cluttered_table.txt")

    # Part D
    print("\nPart D")
    p4d("clean_hallway.txt")

    # Part E
    print("\nPart E")
    p4e("cluttered_hallway.txt")
