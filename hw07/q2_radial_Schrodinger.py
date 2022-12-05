import q1_pendulum as myODE
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime

'''
Set hbar ^ 2 / m as 1 and E is in the unit of hbar ^ 2 / m
e ^ 2 / 4\pi \epsilon_0 is also 1
'''


def V1(r):
    return -1 / r


def V2(r):
    r0 = r / 0.4
    v1 = -3 / r * scipy.special.erf(r0 / np.sqrt(2))
    v2 = np.exp(-r0 ** 2 / 2) * (-14.0093922 + 9.5099073 * r0**2 -
                                 1.7532723 * r0**4 + 0.0834586 * r0**6)

    return v1 + v2


def radial_Schrodinger(y, r, V, E, l):
    dy1 = y[1]
    dy2 = (-2 * E + 2 * V(r) + l * (l + 1) / r ** 2) * y[0]
    return np.array([dy1, dy2])


def plot_u_inf_to_E(res, Es, l=0, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(Es, np.zeros(Es.shape[0]), ls=':', c='gray', alpha=0.5)
        ax.set_xlabel('energy')
        ax.set_ylabel('$\psi_{\infty}$')

    ax.plot(Es, res, label=label)
    ax.legend()
    fig = ax.get_figure()
    ax.set_ylim(-1, 1)
    # plt.show()
    return fig, ax


def solve_potential_brute(l, V=V1):
    # try different energy (eigenvalue)
    res = np.array([])
    Es = np.linspace(-5, 0, 100)
    for E in Es:
        res = np.append(res, get_psi_inf(E, l, r_inf=10, V=V))
    fig, ax = plot_u_inf_to_E(res, Es, l, 'inf = 10')

    for r_inf in [50, 100]:
        res = np.array([])
        for E in Es:
            res = np.append(res, get_psi_inf(E, l, r_inf=r_inf, V=V))
        fig, ax = plot_u_inf_to_E(res, Es, l=l, label=f'inf = {r_inf}', ax=ax)
    # plt.show()
    fig.savefig(f'q2_uinf_to_energy_l={l}_pot2.png')
    # TODO: try more l


def solve_potential_bisec(el, er, V, l=0, r_inf=100, rev=False):
    if not rev:
        psi_l = get_psi_inf(el, l, r_inf, V=V)
        psi_r = get_psi_inf(er, l, r_inf, V=V)
    else:
        psi_l = get_psi_zero(el, l, r_inf, V=V)
        psi_r = get_psi_zero(er, l, r_inf, V=V)
    print('init:', psi_l, psi_r)
    assert (psi_l < 0 and psi_r > 0) or (
        psi_l > 0 and psi_r < 0), 'bad init sect'
    psi_mid = 1
    while np.abs(el - er) > 1e-6 and np.abs(psi_mid) > 1e-5:
        e_mid = (el + er) / 2
        if not rev:
            psi_mid, solver = get_psi_inf(
                e_mid, l, r_inf, V=V, ret_solver=True)
        else:
            psi_mid, solver = get_psi_zero(
                e_mid, l, r_inf, V=V, ret_solver=True)
        if (psi_mid < 0 and psi_l < 0) or (psi_mid > 0 and psi_l > 0):
            el = e_mid
        else:
            er = e_mid
        print(e_mid, psi_mid)
    # plot_psi([solver])
    # plt.show()
    return e_mid, solver


def solve_potential_1_intersec(el, er):
    psi_l = get_psi_inf(el, 0, 100)
    psi_r = get_psi_inf(er, 0, 100)
    print('init:', psi_l, psi_r)
    assert (psi_l < 0 and psi_r > 0) or (
        psi_l > 0 and psi_r < 0), 'bad init sect'
    while np.abs(el - er) > 1e-5:
        e_next = (psi_r * el - psi_l * er) / (psi_r - psi_l)
        psi_next = get_psi_inf(e_next, 0, 100)
        if (psi_next < 0 and psi_l < 0) or (psi_next > 0 and psi_l > 0):
            el = e_next
        else:
            er = e_next

    return e_next


def get_psi_zero(E, l, r_inf=100., V=V1, ret_solver=False):
    def f(x, y):
        return radial_Schrodinger(y, x, V, E, l)
    # Is the boundary condition at r = 0 determined by potential???
    # arbitrary y'_0: take 1
    solver = myODE.ODE(r_inf, np.array([0, 1], dtype=float), f)
    solver.solve_RK4(0, r_step, rev=True)
    # plt.plot(solver.get_res_t(), solver.get_res_theta())
    # plt.show()
    if ret_solver:
        return solver.get_res_theta()[-1], solver
    return solver.get_res_theta()[-1]


def get_psi_inf(E, l, r_inf=100, V=V1, ret_solver=False):
    def f(x, y):
        return radial_Schrodinger(y, x, V, E, l)
    # Is the boundary condition at r = 0 determined by potential???
    # arbitrary y'_0: take 1
    solver = myODE.ODE(0, np.array([0, 1], dtype=float), f)
    solver.solve_RK4(r_inf, r_step)
    # plt.plot(solver.get_res_t(), solver.get_res_theta())
    # plt.show()
    if ret_solver:
        return solver.get_res_theta()[-1], solver
    return solver.get_res_theta()[-1]


def plot_psi(solvers, labels=None, cs=None, scales=None, lses=None):
    assert labels is not None
    fig, ax = plt.subplots(1, 1)
    for solver, label, c, scale, ls in zip(solvers, labels, cs, scales, lses):
        ax.plot(solver.get_res_t(), solver.get_res_theta() * scale,
                label=label, c=c, ls=ls)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('r / a.u.')
    ax.set_ylabel('psi / a.u.')
    ax.legend()
    plt.show()
    return fig, ax


if __name__ == '__main__':
    r_step = 1e-2
    solve_potential_brute(0, V=V2)
    solve_potential_brute(1, V=V2)
    solve_potential_brute(2, V=V2)

    # e1, solver1 = solve_potential_bisec(-0.9, -0.4, V1, l=0, r_inf=100.)
    # e2, solver2 = solve_potential_bisec(-0.2, -0.1, V1, l=0, r_inf=100.)
    # e3, solver3 = solve_potential_bisec(-0.06, -0.05, V1, l=0, r_inf=100.)

    # e1_rev, solver1_rev = solve_potential_bisec(
    #     -0.9, -0.4, V1, l=0, r_inf=100., rev=True)
    # e2_rev, solver2_rev = solve_potential_bisec(
    #     -0.2, -0.1, V1, l=0, r_inf=100., rev=True)
    # e3_rev, solver3_rev = solve_potential_bisec(
    #     -0.06, -0.05, V1, l=0, r_inf=100., rev=True)
    # print(e1, e2, e3, e1_rev, e2_rev, e3_rev)
    # labels = [f'{i}-th state' for i in range(3)] + \
    #     [f'{i}-th state: solve from inf' for i in range(3)]
    # fig, ax = plot_psi([solver1, solver2, solver3, solver1_rev, solver2_rev,
    #                    solver3_rev], labels=labels, cs=['orange', 'steelblue', 'darkgreen'] * 2, scales=[1] * 3 + [1e-4] * 3, lses=['--'] * 3 + ['-'] * 3)
    # fig.savefig('pot_1.png')
