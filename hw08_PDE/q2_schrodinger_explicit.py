import numpy as np
import matplotlib.pyplot as plt

'''
set hbar as 1 and mass as 1
'''


def construct_matrix_explicit(V, x, dt):
    x_grid = x.shape[0]
    dx = (x[-1] - x[0]) / x_grid
    # should I ignore the boundary condition and hope that it will be kept because of the initial condition satisfies it?
    # Or is the Dirichlet condition here automatically satisfied by the truncation of the matrix?
    # what about other types of boundary conditions? say, a non-zero constant.
    # can I, after each iteration, adjust the boundary?
    tmp = np.zeros((x_grid, x_grid))

    for i in range(0, x_grid):
        tmp[i, i] = -2 / dx ** 2 - 2 * V(x[i])
        if i > 0:
            tmp[i, i - 1] = 1 / dx ** 2
        if i < x_grid - 1:
            tmp[i, i + 1] = 1 / dx ** 2
    A = 1j * dt * tmp
    return A


def norm(psi, x):
    dx = (x[-1] - x[0]) / (x.shape[0] - 1)
    return np.sum(np.abs(psi) ** 2 * dx)


def norm_range(psi, x, x1, x2):
    '''returns the portion of wave function between x1 and x2'''
    assert x[0] <= x1 <= x2 <= x[-1]
    lindex = 0
    while x[lindex] < x1:
        lindex += 1
    rindex = lindex
    while x[rindex] < x2:
        rindex += 1
    return norm(psi[lindex: rindex], x[lindex: rindex])


def solve(construct, psi, x, V, t_max, dt):
    t = 0
    # rows: time; columns: x
    sol = np.zeros((int((t_max) / dt) + 1, x.shape[0]))
    last_psi = psi
    iter = 0
    portions = []
    while t < t_max:
        tmp_last = last_psi
        last_psi = psi
        psi = construct(V, x, dt) @ psi + tmp_last
        print("psi at boundary:", psi[0], psi[-1])
        psi[0] = psi[-1] = 0
        # norm shifts and we can manually normalize it
        print("norm:", norm(psi, x))
        psi *= np.sqrt(1 / norm(psi, x))
        left = norm_range(psi, x, 1, x[-1])
        right = norm_range(psi, x, x[0], -1)
        in_well = norm_range(psi, x, -1, 1)
        portions.append([left, right, in_well])
        print("left to the well", right)
        print("right to the well", left)
        print("inside the well", in_well)
        print('\n')
        sol[iter, :] = psi
        iter += 1
        t += dt
    return sol, portions


def get_gaussian(x, x0, p):
    a = 1
    return 1 / (np.sqrt(a / 2) * (2 * np.pi) ** 0.25) * np.exp(1j * p * x - (x - x0) ** 2 / (a ** 2))


def V(x):
    if x < 1 and x > -1:
        return -1
    else:
        return 0


def energy(psi, x):
    dx = x[1] - x[0]
    E = 0
    for i in range(1, psi.shape[0] - 1):
        E += -1 / 2 * (psi[i + 1] + psi[i - 1] - 2 * psi[i]
                       ) / 2 * psi[i] * dx + V(x[i]) * psi[i]**2 * dx
    return E


if __name__ == '__main__':
    t_max = 20
    dt = 1e-3
    p = 1
    x0 = -5
    x = np.linspace(-30, 30, 1000)
    psi0 = get_gaussian(x, x0, p)
    # print(norm(psi0, x))
    # plt.plot(x, psi0)
    # plt.show()
    if input('re-calculate?') in ['y', 'Y']:
        sol, portions = solve(construct_matrix_explicit, psi0, x, V, t_max, dt)
        np.savetxt('solution_explicit.txt', sol)
        np.savetxt('portions_explicit.txt', portions)

    sol = np.loadtxt('solution_explicit.txt', dtype='complex')
    portions = np.loadtxt('portions_explicit.txt', dtype='float')

    print(energy(psi0, x))
    print(energy(sol[-2, :], x))

    fig, ax = plt.subplots(1, 1)

    s = ax.imshow(np.abs(sol) ** 2, extent=[x[0], x[-1], t_max, 0], aspect=10)
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.colorbar(s)
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.linspace(0, t_max, len(portions)),
            [p[0] for p in portions], label='left')
    ax.plot(np.linspace(0, t_max, len(portions)),
            [p[1] for p in portions], label='right')
    ax.plot(np.linspace(0, t_max, len(portions)),
            [p[2] for p in portions], label='inside')
    ax.legend()
    plt.show()
