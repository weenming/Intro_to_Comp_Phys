from operator import ge
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def get_b_recursive(data_pairs):
    if len(data_pairs) == 1:
        return data_pairs[0][1]
    else:
        dy = get_b_recursive(data_pairs[1:]) - get_b_recursive(data_pairs[:-1])
        dx = data_pairs[-1][0] - data_pairs[0][0]
        return dy / dx


def Newton(data_pairs):
    '''
    input data: python list, [[x1, y1], [x2, y2], [x3, y3]...]
    '''
    def interpolation_func(x):
        f = 0
        for n in range(len(data_pairs), 0, -1):
            # I will do it super inefficiently...
            f += get_b_recursive(data_pairs[: n])
            if n > 1:
                f *= x - data_pairs[n - 2][0]
        return f
    return interpolation_func


def Spline(x, y):
    n = x.shape[0]
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    for i in range(n):
        if i == 0 or i == n - 1:
            A[i, i] = 1
        else:
            A[i, i - 1: i +
                2] = np.array([x[i] - x[i-1], 2*(x[i+1]-x[i-1]), x[i+1]-x[i]])
            b[i] = 6*(y[i+1]-y[i])/(x[i+1]-x[i]) + \
                6*(y[i-1]-y[i])/(x[i]-x[i-1])
    d2x = np.linalg.solve(A, b)

    def f(x_plot):
        for i in range(x.shape[0]):
            if x[i] <= x_plot <= x[i + 1]:
                return d2x[i]/6/(x[i+1]-x[i])*(x[i+1]-x_plot)**3 + d2x[i+1]/6/(x[i+1]-x[i])*(x_plot-x[i])**3\
                    + (y[i]/(x[i+1]-x[i]) - d2x[i]*(x[i+1]-x[i])/6) * (x[i+1] - x_plot)\
                    + (y[i+1]/(x[i+1]-x[i]) - d2x[i+1] *
                       (x[i+1]-x[i])/6) * (x_plot - x[i])
    return f


pts_x = np.linspace(0, np.pi, 10)
pts_y = np.cos(pts_x)
spline = Spline(pts_x, pts_y)
plt.plot(pts_x, [spline(x) for x in pts_x])
plt.show()


def plots():
    # cos
    pts_x = np.linspace(0, np.pi, 10)
    pts = [[x, np.cos(x)] for x in np.linspace(0, np.pi, 10)]
    dense_pts = np.linspace(0, np.pi, 1000)
    f_cos = Newton(pts)
    fig, ax = plt.subplots(1, 1)
    ax.plot(dense_pts, f_cos(dense_pts), label='Newton interpolation')
    ax.scatter(pts_x, np.cos(pts_x), label='sampling points')
    # ax.plot(dense_pts, np.cos(dense_pts), label='cos', ls=':')
    ax.plot(dense_pts, [spline(x)
            for x in dense_pts], label='cubic spline', ls=':')
    ax.legend()
    fig.savefig('q1_cos.png', dpi=600)
    plt.show()

    # compare
    print(max(abs(CubicSpline(pts_x, np.cos(pts_x))(dense_pts) - np.cos(dense_pts))))
    print(max(abs(f_cos(dense_pts) - np.cos(dense_pts))))
    fig, ax = plt.subplots(1, 1)
    ax.plot(dense_pts, f_cos(dense_pts) - np.cos(dense_pts),
            label='error of Newton interpolation')
    # ax.scatter(pts_x, np.cos(pts_x), label='sampling points')
    # ax.plot(dense_pts, np.cos(dense_pts), label='cos', ls=':')
    ax.plot(dense_pts, CubicSpline(pts_x, np.cos(pts_x))(dense_pts) -
            np.cos(dense_pts), label='eoor of cubic spline', ls='-')
    ax.legend()
    fig.savefig('q1_cos_compare.png', dpi=600)
    plt.show()

    # 1/x^2+25

    def f(x): return 1 / (1 + 25 * x ** 2)

    pts_x = np.linspace(-1, 1, 10)
    pts = [[x, f(x)] for x in np.linspace(-1, 1, 10)]
    dense_pts = np.linspace(-1, 1, 1000)
    f_bad = Newton(pts)
    spline2 = Spline(pts_x, f(pts_x))
    fig, ax = plt.subplots(1, 1)
    ax.plot(dense_pts, f_bad(dense_pts), label='Newton interpolation')
    ax.scatter(pts_x, f(pts_x), label='sampling points')
    # ax.plot(dense_pts, np.cos(dense_pts), label='cos', ls=':')
    ax.plot(dense_pts, [spline2(x)
            for x in dense_pts], label='cubic spline', ls=':')
    ax.plot(dense_pts, f(dense_pts), label='precise')
    ax.legend()
    fig.savefig('q1_bad.png', dpi=600)
    plt.show()

    # compare
    print(max(abs((CubicSpline(pts_x, f(pts_x))(dense_pts) - f(dense_pts)))))
    print(max(abs((f_bad(dense_pts) - f(dense_pts)))))
    fig, ax = plt.subplots(1, 1)
    ax.plot(dense_pts, f_bad(dense_pts) -
            f(dense_pts), label='Newton interpolation')
    # ax.scatter(pts_x, f(pts_x), label='sampling points')
    # ax.plot(dense_pts, np.cos(dense_pts), label='cos', ls=':')
    ax.plot(dense_pts, CubicSpline(pts_x, f(pts_x))(
        dense_pts) - f(dense_pts), label='cubic spline', ls='-')
    # ax.plot(dense_pts, f(dense_pts), label='precise')
    ax.legend()
    fig.savefig('q1_bad_compare.png', dpi=600)
    plt.show()


plots()
