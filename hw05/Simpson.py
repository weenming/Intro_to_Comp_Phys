import numpy as np
import matplotlib.pyplot as plt


def psi(x):
    return 2 / 3 / np.sqrt(3) * (1 - 2 / 3 * x + 2 / 27 * x ** 2) * np.exp(-x / 3)


def f(x):
    return psi(x) ** 2 * x ** 2


def simpson_eqspace(f, xmin, xmax, N):
    assert N % 2 == 1, 'must have odd sampling points'
    xs = np.linspace(xmin, xmax, N)
    fs = f(xs)
    res = 0
    h = (xmax - xmin) / N
    for i in range(N - 2):
        res += h / 6 * (fs[i] + 4 * fs[i + 1] + fs[i + 2])
    return res


def simpson_expspace(f, xmin, xmax, N):
    r0 = 0.0005
    ts = np.linspace(0, np.log(xmax / r0 + 1), N)
    rs = r0 * (np.exp(ts) - 1)
    res = 0
    # there's actually 2N - 1 points which is always odd
    for i in range(N - 1):
        l = rs[i]
        r = rs[i + 1]
        res += (r - l) / 6 * (f(r) + 4 * f((r + l) / 2) + f(l))
    return res


if __name__ == '__main__':
    ns = range(1, 250, 2)
    ns0 = range(1, 501, 2)
    eqspace = np.array([simpson_eqspace(f, 0, 40, n) for n in ns0])
    expspace = np.array([simpson_expspace(f, 0, 40, n) for n in ns])
    print(simpson_expspace(f, 0, 40, 50))
    fig, ax = plt.subplots(1, 1)
    ax.plot(ns0, -np.log10(np.abs(1 - eqspace)), label='equal space')
    ax.plot([n * 2 for n in ns], -
            np.log10(np.abs(1 - expspace)), label='exp space')
    ax.legend()
    ax.set_xlabel('number of intervals, n')
    ax.set_ylabel('negative log10 of error')
    fig.savefig('q2_res.png', dpi=600)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, 40, 1000)
    ax.plot(x, np.abs(psi(x)) ** 2, label='$\|\psi_{3s}\|^2$')
    ax.plot(x, x ** 2 * np.abs(psi(x)) ** 2, label='$r^2 \|\psi_{3s}\|^2$')
    ax.set_xlabel('r / a.u.')
    ax.set_ylabel('a.u.')
    ax.legend()
    fig.savefig('q2_demo.png', dpi=600)
