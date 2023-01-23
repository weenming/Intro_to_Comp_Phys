import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# But what influence will it have if the modeled region is lager?


class NormVector3D:
    def __init__(self):
        self.rand_on_sphere()

    def x(self):
        return self.vec[0]

    def y(self):
        return self.vec[1]

    def z(self):
        return self.vec[2]

    '''
    Note the inefficiency of this implementation: about 0.1s for one rand vec
    '''

    def rand_on_sphere(self):
        for_select = np.random.rand(3) * 2 - 1
        for _ in range(int(1e2)):
            this_vec = np.random.rand(3) * 2 - 1
            if self.get_norm(this_vec) < 1:
                for_select = np.vstack((for_select, this_vec))
        row = for_select.shape[0]
        self.vec = for_select[int(np.random.rand() * row), :]
        self.normalize()

    def rand_on_sphere_faster1(self):
        vec = np.random.rand(3) * 2 - 1
        while self.get_norm(vec) > 1:
            vec = np.random.rand(3) * 2 - 1
        self.vec = vec
        self.normalize()

    def normalize(self):
        self.vec /= self.get_norm(self.vec)

    @staticmethod
    def get_norm(vec):
        if isinstance(vec, NormVector3D):
            return np.sqrt(np.sum(np.abs(vec.vec) ** 2))
        else:
            return np.sqrt(np.sum(np.abs(vec) ** 2))

    @staticmethod
    def innerProduct(vec1, vec2):
        return vec1.vec @ vec2.vec


'''
Only 4 atoms in a unit cell that are independent if periodic boundary condition is adopted?
Set k_B as 1
'''


class FCC:
    def __init__(self, N):
        '''
        N * N * N cells
        '''
        self.N = N
        # i, j, k cannot simultaneously be odd number
        self.vertices = np.empty((2 * N, 2 * N, 2 * N), dtype='object')

        def init(i, j, k):
            self.vertices[i, j, k] = NormVector3D()
        self.traverse(init)
        self.H = self.get_H_total()

    def traverse(self, f, *args):
        N = self.N
        for i in range(2 * N):
            for j in range(2 * N):
                for k in range(2 * N):
                    if not self._is_center_or_edge(i, j, k):
                        f(i, j, k, *args)

    def _is_center_or_edge(self, i, j, k):
        # true iff i, j, k are all odd or one of i, j, k is odd
        return (i % 2 + j % 2 + k % 2) % 2

    # MAY CONTAIN BUG!!!

    def get_H_total(self):
        '''
        It is obvious that 1 particle has 12 different NNs.
        '''
        H = 0

        def get_neighbor_H(i, j, k):
            '''
            (+- 1 / 2, 0) ^ 3: 27 neighbors in total, but centers should be neglected
            '''
            nonlocal H
            H += self.get_H_local(i, j, k)
        self.traverse(get_neighbor_H)
        return H

    def get_H_local(self, i, j, k):
        H = 0
        dif = [-1, 0, 1]
        for di in dif:
            ni = i + di
            for dj in dif:
                nj = j + dj
                for dk in dif:
                    nk = k + dk
                    if self._is_center_or_edge(ni, nj, nk) or not (di or dj or dk):
                        # when not a neighbor (itself) also discard
                        continue
                    # adopt periodic boundary condition
                    ni, nj, nk = self._periodic_fix(ni, nj, nk)
                    # every coupling is counted twice
                    H -= NormVector3D.innerProduct(
                        self.vertices[i, j, k], self.vertices[ni, nj, nk])
        return H

    def _periodic_fix(self, i, j, k):
        if i < 0 or i >= 2 * self.N:
            i %= 2 * self.N
        if j < 0 or j >= 2 * self.N:
            j %= 2 * self.N
        if k < 0 or k >= 2 * self.N:
            k %= 2 * self.N
        return i, j, k

    def rand_displace(self, scale=0.1):
        # randomly choose an atom
        i, j, k = np.random.randint(0, 2 * self.N, 3)

        # print('before fix', i, j, k)
        if self._is_center_or_edge(i, j, k):
            i += 1
        i, j, k = self._periodic_fix(i, j, k)
        # store last H
        self.equ_H = self.H
        self.last_ijk = (i, j, k)
        self.last_v = self.vertices[i, j, k].vec.copy()
        # prepare for updating H
        self.H -= self.get_H_local(i, j, k)
        # print('after fix', i, j, k)
        # randomly change direction
        self.vertices[i, j, k].vec += (np.random.rand(3) * 2 - 1)
        self.vertices[i, j, k].normalize()
        # update H by only evaluating neighbors
        self.H += self.get_H_local(i, j, k)

        return

    def revert(self):
        # should be more efficient than copy.deepcopy?
        self.H = self.equ_H
        i, j, k = self.last_ijk
        self.vertices[i, j, k].vec = self.last_v

    def get_avg_s(self):
        self.total_x = 0
        self.total_y = 0
        self.total_z = 0

        def get_s(i, j, k):
            self.total_x += self.vertices[i, j, k].x()
            self.total_y += self.vertices[i, j, k].y()
            self.total_z += self.vertices[i, j, k].z()

        self.traverse(get_s)
        # should divide by 4 N^3
        return np.sqrt(self.total_x ** 2 + self.total_y ** 2 + self.total_z ** 2) / (self.N) ** 3 / 4


def MC_Heisenberg(T, N_size=3, N_step=1000, avg=lambda x: avg(x, 100)):
    res = []
    fcc = FCC(N_size)
    for _ in range(N_step):
        equ_H = fcc.H
        res.append({'H': fcc.H, 's': fcc.get_avg_s()})
        # print(fcc.get_H_total())
        fcc.rand_displace()
        # H updated
        this_H = fcc.H
        if equ_H > this_H:
            # accept
            # print('acc 1')
            pass
        else:
            acc_rate = np.exp(-(this_H - equ_H) / T)
            if np.random.rand() > acc_rate:
                # reject
                # print('rej')
                fcc.revert()
            else:
                # accept
                pass
                # print('acc 2')
    # otherwise memory will soon run out...
    # if sample at 1000 temperature and each run iters 10000 times then it's 1e7 * size of float, which is ~100MB
    this_H = [r['H'] for r in res]
    np.savetxt(f'./res_5/H_5_at_T={T}', this_H)

    return this_H, avg([r['s'] for r in res])


def avg(x, n=100):
    try:
        return np.sum(x[-n: -1]) / (n - 1)
    except:
        return np.sum(x[0: -1]) / len(x)


def s_to_T(cell_number, steps, Ts=np.linspace(1e-100, 10, 100), avg=lambda x: avg(x, 100)):
    # looking for Curie temperature
    equ_H = []
    equ_s = []
    for T in Ts:
        print(T)
        this_H, this_s = MC_Heisenberg(T, cell_number, steps, avg)

        equ_H.append(this_H)
        equ_s.append(this_s)

    return Ts, {'equ_H': equ_H, 'equ_s': equ_s}


def plot(Ts, res, fig=None, ax=None, l='', c_s='orange', c_H='steelblue'):
    equ_H = res['equ_H']
    equ_s = res['equ_s']

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        axT = ax.twinx()
        axT.set_ylabel('total H / a.u.')
        axT.plot(Ts, equ_H, label='H' + l, color=c_H)
        axT.legend(loc='upper right')
    else:
        assert fig is not None
    ax.plot(Ts, equ_s, label='s' + l, color=c_s)
    ax.legend(loc='upper left')
    ax.set_ylabel('|s| / a.u.')
    ax.set_xlabel('T / K')

    return fig, ax


if __name__ == '__main__':
    cell_number = 5
    steps = 80000
    Ts_5, res_5 = s_to_T(cell_number, steps, Ts=np.linspace(0.001, 10, 200))
    fig, ax = plot(Ts_5, res_5)
    np.savetxt('res_5/all_T', Ts_5)
    np.savetxt('res_5/all_s', res_5['equ_s'])
    ax.set_title(
        f'Classical Heisenberg model\n One period: {cell_number}*{cell_number}*{cell_number} cells')
    fig.savefig(f'q2_s_to_T_{cell_number ** 3}cells.png', dpi=300)
