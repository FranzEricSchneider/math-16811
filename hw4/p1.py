import numpy
from matplotlib import pyplot


def dydx(yval):
    return 1 / (2 * yval)


def main():
    # Goes from 2.0 to 1.0
    step = -0.05
    x = numpy.abs(numpy.arange(2, 0.99, step))
    # True y
    # Take abs just in case the final value is negative floating point error
    # and becomes undefined
    y = numpy.sqrt(numpy.abs(x - 1))

    euler_y = euler(dydx, step, num_steps=len(x)-1, y_start=1.0)
    rk4_y = runge_kutta_4(dydx, step, num_steps=len(x)-1, y_start=1.0)
    ab4_y = adams_bashforth_4(dydx, step, num_steps=len(x)-1,
                             y_start=[1.07238052947636,
                                      1.04880884817015,
                                      1.02469507659596,
                                      1.0])

    plot(x,
         ys=[euler_y, rk4_y],
         labels=["Euler Method",
                 "Runge-Kutta 4"],
         title="Euler and Runge-Kutta 4 with the original function")
    plot(x,
         ys=[euler_y, ab4_y],
         labels=["Euler Method",
                 "Adams-Bashforth 4"],
         title="Euler and Adams-Bashforth 4 with the original function")

    for xx, yy, ey, rky, aby in zip(x, y, euler_y, rk4_y, ab4_y):
        print(f"\t\t\t{xx:.2f} & {yy:.4f} & {ey:.4f} & {yy - ey:.8f}"
              f" & {rky:.4f} & {yy - rky:.8f}"
              f" & {aby:.4f} & {yy - aby:.8f} \\\\")


def euler(dydx, step, num_steps, y_start):
    y = [y_start]
    for _ in range(num_steps):
        y.append(y[-1] + step * dydx(y[-1]))
    return numpy.array(y)


def runge_kutta_4(dydx, step, num_steps, y_start):
    y = [y_start]
    for _ in range(num_steps):
        # TODO: OH: Should X be involved here somewhere?
        # TODO: OH: Maybe a negatvie step doesn't work?
        k1 = step * dydx(y[-1])
        k2 = step * dydx(y[-1] + k1/2)
        k3 = step * dydx(y[-1] + k2/2)
        k4 = step * dydx(y[-1] + k3)
        y.append(y[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6)
    return numpy.array(y)


def adams_bashforth_4(dydx, step, num_steps, y_start):
    y = y_start
    for i in range(num_steps):
        j = i + 3
        polynomial = (step/24) * (55*dydx(y[j]) -
                                  59*dydx(y[j-1]) +
                                  37*dydx(y[j-2]) -
                                  9*dydx(y[j-3]))
        y.append(y[-1] + polynomial)
    return numpy.array(y)[3:]


def plot(x, ys, labels, title):
    true_x = numpy.linspace(x.min(), x.max(), 200)
    true_y = numpy.sqrt(numpy.abs(true_x - 1))
    pyplot.plot(true_x, true_y, "-", label="True y", lw=3)
    for y, label in zip(ys, labels):
        pyplot.plot(x, y, "o", label=label)
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
