import numpy as np
import matplotlib.pyplot as plt


def rho(x, y):
    return 1


def rho_zero(x, y):
    return 0


def b1(name, val):
    if name == 'y' and val == Ly:
        return 1
    else:
        return 0


def b2(x, y):
    return 0


def construct_matrix(Lx, Ly, b, rho):
    x_grid = 100
    y_grid = 100
    # u is arranged in the following way:
    # u11, u12, u13, ..., u1-98, u21, u22, ..., ..., u98-1, u98-2, ..., u_98-98
    # u[0], ...u[2], ..., u[97], u[98], ...        , u[98 * 97 + 0],.., u[98 * 97 + 97]
    A = np.zeros(x_grid - 2, y_grid - 2)
    x = np.linspace(0, Lx, x_grid)
    y = np.linspace(0, Ly, y_grid)
    for i in range(x_grid):
        for j in range(y_grid):
            if i == j:
                A[i, j] += 4
            if (abs(j - i) == 1 or abs(j - i) == 4):
                A[i, j] -= 1
    # boundary conditions


def error(x1, x2):
    return np.amax(abs(x1 - x2))


def relaxation_method(Lx, Ly, boundary_condition, rho):
    '''
    Solves Poission equations using the relaxation method (Gauss-Seidel method)

    In the range of x \in (0, Lx) and y \in (0, Ly)
    Form of the equation is Au =
    '''

    # A, b = construct_matrix(Lx, Ly, boundary_condition, rho)
    # sol = solve_relaxation(A, b)

    # use stupid iteration instead because constructing the matrix is tedious :(

    x_grid = 100
    y_grid = 100
    x = np.linspace(0, Lx, x_grid)
    y = np.linspace(0, Ly, y_grid)
    h_sq = Ly * Lx / (x_grid * y_grid)
    u = np.zeros((x_grid, y_grid))
    # x, 0
    u[0, 1:x_grid - 1] = boundary_condition('y', 0)
    # x, Ly
    u[y_grid - 1, 1:x_grid - 1] = boundary_condition('y', Ly)
    # y, 0
    u[1:y_grid - 1, 0] = boundary_condition('x', 0)
    # y, Lx
    u[1:y_grid - 1, Lx] = boundary_condition('x', Lx)

    last_u = u.copy() + 1
    tol = 1e-5
    iter = 0
    while error(u, last_u) > tol:
        print(f"iter {iter}")
        iter += 1
        # starting from the lower left corner
        last_u = u.copy()
        for i in range(1, x_grid - 1):
            for j in range(1, y_grid - 1):
                u[i, j] = 1 / 4 * (last_u[i + 1, j] + last_u[i, j + 1] +
                                   u[i - 1, j] + u[i, j - 1] + h_sq * rho(x[i], y[j]))
    return u


if __name__ == '__main__':
    # q1.1 (~2000 iters for tol = 1e-5)
    # Lx, Ly = 1, 1.5
    # sol = relaxation_method(Lx, Ly, b1, rho_zero)

    # q1.2 (~4000 iters for tol = 1e-5)
    Lx, Ly = 1, 1
    sol = relaxation_method(Lx, Ly, b2, rho)

    fig, ax = plt.subplots(1, 1)

    s = ax.imshow(sol, extent=[0, Lx, Ly, 0])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    cbar = fig.colorbar(s)
    cbar.set_label('$\phi / V$')

    plt.show()
