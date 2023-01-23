import numpy as np
import matplotlib.pyplot as plt
import time

g = 9.8
l = 1


def timer(func):
    def time_the_func(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('this function takes', end - begin, 'seconds')
    return time_the_func


class ODE():
    def __init__(self, x0, y0: np.array, f):
        '''
        eq: {d\vec{y} \over dx} = \vec{f}(\vec{y})
        '''
        assert y0.shape[0] > 0, 'must specify init val'
        assert len(y0.shape) == 1 or y0.shape[1] == 0, 'y is 1-dim'
        self._y0 = self._y = y0
        self._x0 = self._x = x0
        self._f = f
        self.res = None
        return

    @timer
    def solve_Euler(self, x_stop, x_step):
        self.res = []
        while (self._x < x_stop):
            self._x += x_step
            self._y += self._f(self._x, self._y) * x_step
            # print(self._y)
            self.res.append([self._x, self._y.copy()])
        return

    @timer
    def solve_Euler_mod(self, x_stop, x_step):
        '''
        estimate with averaged slope
        '''
        self.res = []
        while (self._x < x_stop):
            self._x += x_step
            tmp_dy = self._f(self._x, self._y) * x_step
            tmp_y = self._y + tmp_dy
            self._y += (tmp_dy + self._f(self._x, tmp_y) * x_step) / 2
            # print(self._y)
            self.res.append([self._x, self._y.copy()])
        return

    @timer
    def solve_mid_point(self, x_stop, x_step):
        self.res = []
        while (self._x < x_stop):
            self._x += x_step
            tmp_dy = self._f(self._x, self._y) * x_step
            mid_y = self._y + tmp_dy / 2
            # here f does not rely explicitly on x (time)
            self._y += self._f(self._x, mid_y) * x_step
            # print(self._y)
            self.res.append([self._x, self._y.copy()])
        return

    @timer
    def solve_ET(self, x_stop, x_step, err_tol=1e-10):
        assert err_tol > 0
        self.res = []
        h = x_step
        f = self._f
        while self._x < x_stop:

            # y_istar = self._y
            last_corr_y = self._y
            corr_y = last_corr_y + err_tol * 2
            init_dy = f(self._x, self._y) * x_step
            while np.abs((last_corr_y - corr_y)).sum() > err_tol:
                last_corr_y = corr_y
                last_corr_dy = f(self._x, last_corr_y) * x_step
                corr_y = self._y + (last_corr_dy + init_dy) / 2
            self._y = corr_y
            self._x += x_step
            self.res.append([self._x, self._y.copy()])
        return

    @timer
    def solve_RK4(self, x_stop, x_step, rev=False):
        self.res = []
        h = x_step
        f = self._f
        while (self._x < x_stop):
            # here f does not rely explicitly on x (time)
            k1 = f(self._x, self._y)
            k2 = f(self._x + h / 2, self._y + 1 / 2 * k1 * h)
            k3 = f(self._x + h / 2, self._y + 1 / 2 * k2 * h)
            k4 = f(self._x + h, self._y + k3 * h)

            self._y += (k1 + 2 * k2 + 2 * k3 + k4) / 6 * x_step
            # print(self._y)
            self._x += h
            self.res.append([self._x, self._y.copy()])
        if rev:
            while (self._x > x_stop):
                # here f does not rely explicitly on x (time)
                k1 = f(self._x, self._y)
                k2 = f(self._x - h / 2, self._y - 1 / 2 * k1 * h)
                k3 = f(self._x - h / 2, self._y - 1 / 2 * k2 * h)
                k4 = f(self._x - h, self._y - k3 * h)

                self._y -= (k1 + 2 * k2 + 2 * k3 + k4) / 6 * x_step
                # print(self._y)
                self._x -= h
                self.res.append([self._x, self._y.copy()])
        return

    def get_res_t(self):
        return np.array([res_i[0] for res_i in self.res])

    def get_res_omega(self):
        return np.array([res_i[1][1] for res_i in self.res])

    def get_res_theta(self):
        return np.array([res_i[1][0] for res_i in self.res])

    def calc_energy(self, plot=False):
        omega = np.array(self.get_res_omega())
        theta = np.array(self.get_res_theta())
        energy = (1 - np.cos(theta)) * g * l + l ** 2 / 2 * omega ** 2
        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.get_res_t(), energy)
            plt.show()
        return energy


def pendulum_motion(x, y: np.array):
    '''
    y1: \theta
    y2: \dot{\theta}
    '''
    return np.array([y[1], -g / l * np.sin(y[0])])


def plot_omega_and_theta(solver: ODE):
    if solver.res is None:
        return None, None

    t = [res_i[0] for res_i in solver.res]
    theta = [res_i[1][0] for res_i in solver.res]
    omega = [res_i[1][1] for res_i in solver.res]
    fig, ax = plt.subplots(1, 1)
    l1 = ax.plot(t, theta, label='$\\theta$')
    ax.set_ylabel('$\\theta$')
    ax.set_xlabel('time / s')
    ax.grid(ls=":")
    ax2 = ax.twinx()
    l2 = ax2.plot(t, omega, label='$\\omega$', color='orange')
    ax2.set_ylabel('$\\omega$')

    ls = l1 + l2
    labels = [l.get_label() for l in ls]
    ax.legend(ls, labels, loc=0)

    plt.show()
    return fig, ax


def plot_energy(solver1, solver2, solver3, solver4, solver5):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    ax1.plot(solver3.get_res_t(), solver3.calc_energy(), label='RK4 method')
    ax1.legend()
    ax1.set_xlabel('time / s')
    ax1.set_ylabel('energy / a.u.')

    ax2.plot(solver2.get_res_t(), solver2.calc_energy(),
             label='midpoint method')
    ax2.plot(solver3.get_res_t(), solver3.calc_energy(), label='RK4 method')
    ax2.plot(solver4.get_res_t(), solver4.calc_energy(),
             label='Euler-trapezoidal method')
    ax2.plot(solver5.get_res_t(), solver5.calc_energy(),
             label='modified Euler method')
    ax2.legend()
    ax2.set_xlabel('time / s')
    ax2.set_ylabel('energy / a.u.')
    plt.show()
    # fig.savefig('energy.png')


if __name__ == '__main__':
    x_start = 0
    x_stop = 20
    x_step = 1e-3
    init_theta = np.pi / 4 * 3
    init_omega = 0

    # # # Euler method
    solver1 = ODE(x_start, np.array([init_theta, init_omega]), pendulum_motion)
    solver1.solve_Euler(x_stop, x_step)
    # fig, _ = plot_omega_and_theta(solver1)
    # fig.savefig('omega_and_theta_Euler.png')
    solver1.calc_energy(plot=False)

    # midpoint method
    solver2 = ODE(x_start, np.array([init_theta, init_omega]), pendulum_motion)
    solver2.solve_mid_point(x_stop, x_step)
    # fig, _ = plot_omega_and_theta(solver2)
    # fig.savefig('omega_and_theta_midpoint.png')
    solver2.calc_energy(plot=False)

    # RK4 method
    solver3 = ODE(x_start, np.array(
        [init_theta, init_omega]), pendulum_motion)
    solver3.solve_RK4(x_stop, x_step)
    # fig, _ = plot_omega_and_theta(solver3)
    # fig.savefig('omega_and_theta_RK4.png')
    solver3.calc_energy(plot=False)

    # ET method
    solver4 = ODE(x_start, np.array([init_theta, init_omega]), pendulum_motion)
    solver4.solve_ET(x_stop, x_step, err_tol=1e-15)
    # fig, _ = plot_omega_and_theta(solver4)
    # fig.savefig('omega_and_theta_Euler-trabalabala.png')
    solver4.calc_energy(plot=False)

    # modified Euler method
    solver5 = ODE(x_start, np.array([init_theta, init_omega]), pendulum_motion)
    solver5.solve_Euler_mod(x_stop, x_step)
    # fig, _ = plot_omega_and_theta(solver5)
    # fig.savefig('omega_and_theta_modified_euler.png')
    solver5.calc_energy(plot=False)

    # plot_energy(solver1, solver2, solver3, solver4, solver5)
    fig, ax = plt.subplots(1, 1)
    ax.plot(solver1.get_res_t(), solver1.calc_energy(), label='Euler method')
    # ax.plot(solver4.get_res_t(), solver4.calc_energy(),
    # label='Euler-trape. method')
    ax.legend()
    plt.show()
