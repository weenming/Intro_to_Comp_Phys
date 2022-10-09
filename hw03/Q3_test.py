import Q3_schrodinger as q3
import matplotlib.pyplot as plt
import numpy as np


def test_convergence():
    a = [q3.basis_wid(w) for w in [1, 2, 3]]
    x = np.linspace(-10, 10, 1000)
    innerporducts = [q3.inner_product(a[0], a[0], lower=-i, upper=i, step=1e-3)
                     for i in [0.5, 1, 5, 10, 20, 30, 50, 100, 200, 1000, 10000]]
    print(innerporducts)
    plt.plot(x, a[0](x))
    plt.show()


def test_with_harmonic_base_state():
    # in harmonic potential, omega = 1 while other constants are all 1
    def psi(x): return (1/np.pi) ** 0.25 * \
        np.exp(-1 / 2 * x ** 2)


    x = np.linspace(-5, 5, 1000)
    plt.plot(x, psi(x))
    plt.plot(x, q3.hamilton_operator(psi, lambda x: x**2 / 2)(x))
    plt.show()
    print(q3.inner_product(psi, q3.hamilton_operator(
        psi, lambda x: x ** 2 / 2), step=1e-2))
    print(q3.inner_product(psi, psi))
    # print(q3.hamilton_operator(lambda x: x ** 2, lambda x: 0)(x))


def test_normalization(w):
    def psi(x):
        return (2 * w/np.pi) ** 0.25 * \
        np.exp(-1 *w * x ** 2)
    print(q3.inner_product(psi, psi))

def test_multiple_return_func():
    x = np.linspace(-20, 20, 300)
    # 这个图可以看出来那个列表里存的函数内容是一样的，虽然地址不一样
    print(q3.basis_wids([0.5, 0.1]))
    plt.plot(x, q3.basis_wids([0.5, 1])[0](x))
    plt.plot(x, q3.basis_wids([0.5, 1])[1](x))
    plt.show()

def test_sqrt():
    a = np.array([[ 0.75592895,  1.13389342], [ 0.37796447,  1.88982237]])

    print(q3.matrix_sqrt(a))

test_normalization(46285673.)