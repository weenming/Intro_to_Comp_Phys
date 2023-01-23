import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

x = np.linspace(1.0, 9.0, 9)
T = np.array([14.6, 18.5, 36.6, 30.8, 59.2, 60.1, 62.2, 79.4, 99.9])


def fit_linear(x, y):
    assert x.shape == y.shape, 'input data must align'
    print(x.shape)
    n = x.shape[0]
    A = np.zeros((n, 2)) + 1
    A[:, 0] = x
    print(A)
    # find least square coefficients
    b0, b1 = inv(A.transpose()@A)@A.transpose()@y
    return b0, b1


def fit_parabolic(x, y):
    assert x.shape == y.shape, 'input data must align'
    print(x.shape)
    n = x.shape[0]
    A = np.zeros((n, 3)) + 1
    A[:, 1] = x
    A[:, 0] = x ** 2
    print(A)
    # find least square coefficients
    b0, b1, b2 = inv(A.transpose()@A)@A.transpose()@y
    return b0, b1, b2


a, b = fit_linear(x, T)
print(f'linear: {a}, {b}]')
def linear(x): return a * x + b


a1, b1, c1 = fit_parabolic(x, T)
print(f'parabolic fit: {a1, b1, c1}')
def parabolic(x): return a1 * x ** 2 + b1 * x + c1


x_to_plot = np.linspace(0, 10, 100)
fig, ax = plt.subplots(1, 1)
ax.plot(x_to_plot, linear(x_to_plot), label='linear fit')
ax.plot(x_to_plot, parabolic(x_to_plot), label='parabolic fit')
ax.scatter(x, T, label='data to fit', color='green')
ax.legend()
ax.set_xlabel('$x$'+'/cm')
ax.set_ylabel('$T$'+'/'+'$^\circ$'+'C')
fig.savefig('q2.png', dpi=600)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(x, (linear(x) - T) ** 2, label='linear fit')
ax2.bar(x, (parabolic(x) - T) ** 2, label='parabolic fit')
# ax.scatter(x, T, label='data to fit', color='green')
# ax1.legend()
# ax2.legend()
# ax2.sharex(ax1)
ax1.set_title('linear fit')
ax2.set_title('parabolic fit')

ax2.sharey(ax1)
ax1.set_xlabel('$x$'+'/cm')
ax2.set_xlabel('$x$'+'/cm')
ax1.set_ylabel('$(\Delta T)^2$'+'/'+'$^\circ$'+'C')
fig.savefig('q2_residual.png', dpi=600)
plt.show()
