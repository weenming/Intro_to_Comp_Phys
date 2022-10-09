from locale import normalize
from os import stat
from re import S
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# if these constants are changed, params in psi need to change, as well
hbar = 1.
m = 1.


def hamilton_operator(wave_func, v):
    def Hpsi(x):
        diff2 = np.zeros(x.shape[0])
        for i in range(x.shape[0] - 2):
            diff2[i + 1] = (wave_func(x[i + 2]) + wave_func(x[i]) -
                            2 * wave_func(x[i + 1])) / (x[i + 2] - x[i]) ** 2 * 4
        diff2[0] = diff2[-1] = diff2[-2]
        potential = v(x) * wave_func(x)
        return potential - diff2 * hbar ** 2 / (2 * m)
    return Hpsi


def inner_product(func1, func2, lower=-10, upper=10, step=1e-3):
    x = np.linspace(lower, upper, int((upper - lower) / step))
    return np.dot(func1(x), func2(x)) * step


def calc_hamilton(basis):
    '''
    basis: python list of wave functions
    '''
    print('start calculating H')
    l = len(basis)
    hamilton = np.zeros((l, l))
    for i in tqdm(range(l)):
        for j in tqdm(range(l), leave=False):
            hamilton[i, j] = inner_product(
                basis[i], hamilton_operator(basis[j], V1))
            
    return hamilton


def calc_s(basis):
    print('start calculating s')
    l = len(basis)
    s = np.zeros((l, l))
    for i in tqdm(range(l)):
        for j in tqdm(range(l), leave=False):
            s[i, j] = inner_product(basis[i], basis[j])
    return s


def V1(x):
    # harmonic oscillator, omega = \sqrt{2}
    return x ** 2


def basis_pos(centers):
    width = 1
    psis = []
    for center in centers:
        def psi(x):
            return np.sqrt(width / np.pi) * np.exp(-width * (x - center) ** 2)
        psis.append(psi)
    return psis


def basis_wids(ws):
    '''
    ws characterizes how 'sharp' the wave function is
    '''
    center = 0
    psis = []
    for w in ws:
        # 我之前是这么写的：def psi: return xxx 然后 psis.append(psi)
        # 结果不行（这是因为在list里存的函数在被调用的时候回这里找只能看到最后一次定义的那个吗）
        # 然后改成用lambda表达式这样写，但是也不行，纯看不懂了
        psis.append(lambda x: np.sqrt(w / np.pi) * np.exp(- w * (x - center) ** 2))
    return psis

def basis_wid(w, odd=False):
    '''
    ws characterizes how 'sharp' the wave function is
    '''
    center = 0
    if odd:
        return lambda x: x * (2 * w / np.pi) ** 0.25 * np.exp(- w * (x - center) ** 2)

    return lambda x: (2 * w / np.pi) ** 0.25 * np.exp(- w * (x - center) ** 2)

def matrix_sqrt(s):
    '''
    return S^{1/2}, S^{-1/2}
    '''
    D, U = np.linalg.eigh(s)
    D = np.diag(D)
    D_sqrt = np.sqrt(D)
    S_sqrt_pos = np.dot(np.dot(U, D_sqrt), np.linalg.inv(U))
    S_sqrt_neg = np.dot(np.dot(U, np.linalg.inv(D_sqrt)), np.linalg.inv(U))
    return S_sqrt_pos, S_sqrt_neg


def get_eigenstates(C, basis, j):
    def eigenstate(x):
        res = 0
        for i in range(len(basis)):
            res += C[i, j] * basis[i](x)
        return res
    return eigenstate


def normalize(C, S):
    for i in range(C.shape[0]):
        C[:, i] /= np.sqrt(C[:, i] @ (S @ C[:, i]))
    return C


def plot_wave_func(states, E):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 9)
    x = np.linspace(-10, 10, 1000)
    for i in range(len(states)):
        if i % 4 == 0 and i < 10:
            ax.plot(x, np.power(states[i](x), 2), label=f'{i+1}-th wave function with energy {E[i] / np.sqrt(2)}'+'$\cdot\sqrt{2}$')
    ax.set_xlabel('x')
    ax.set_ylabel('$|\psi|$')
    ax.legend()
    ax.set_title(f'total {len(states)} basis')
    fig.savefig(f'wave functions with {len(states)} basis', dpi=600)
    plt.show()


def main_width_dif_basis_num(basis_num, is_odd):
    # 这样就不行，算出来的全是同一个。原因是lazy evaluation，
    # 在调用basis[x]的时候回到原来scope里，psi里的w全都是for循环结束前的最后一个
    # basis = basis_wids(np.linspace(0.1, 1, 10))
    # 这样就可以通过多次调用函数创造不同的scope
    #larger param w, wider basis wave function
    basis = [basis_wid(w, is_odd) for w in np.linspace(0.1, 1, basis_num)]
    H = calc_hamilton(basis)
    S = calc_s(basis)

    np.savetxt(f'H_basis_num{len(basis)}', H)
    np.savetxt(f'S_basis_num{len(basis)}', S)
    # H = np.loadtxt('H')
    # S = np.loadtxt('S')

    # eigenvalue problem: H'C' = EC', where H' = S^{-1/2}HS^{1/2}, C' = S^{1/2}C
    S_sqrt_pos, S_sqrt_neg = matrix_sqrt(S)
    H_primed = S_sqrt_neg.dot(H).dot(S_sqrt_neg)
    E, C_primed = np.linalg.eig(H_primed)

    print(E)
    C = np.dot(S_sqrt_neg, C_primed)

    C_normalized = normalize(C, S)
    # notice that \sum_i C_i \psi_i (\psi_i is the i-th basis) cannot be normalized directly
    # because there are cross terms as in matrix S.

    eigen_states_normalized = [get_eigenstates(C_normalized, basis, j) for j in range(len(basis))]
    
    # sort energy and states by E (sorted in ascending order)
    eigen_states_normalized = [x for _, x in sorted(zip(E, eigen_states_normalized), reverse=False)]
    E.sort()

    # plot and save figure
    # plot_wave_func(eigen_states_normalized, E)

    print('check normalization:')
    print([inner_product(eigen_states_normalized[i], eigen_states_normalized[i]) for i in range(len(basis))])

    return E
    

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([2, 11], [y, y], ls=':', color='grey', label='$E={y}\hbar\omega$', zorder=0)
    for basis_num in [3, 4, 5, 6, 7, 8, 9, 10]:
        this_E = main_width_dif_basis_num(basis_num, is_odd=False)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E[0] / np.sqrt(2), color='orange', marker='.', zorder=10)
        ax.scatter([basis_num], this_E[1] / np.sqrt(2), color='orange', marker='x', zorder=10)
        ax.scatter([basis_num], this_E[2] / np.sqrt(2), color='orange', marker='+', zorder=10)

        this_E = main_width_dif_basis_num(basis_num, is_odd=True)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E[0] / np.sqrt(2), color='steelblue', marker='.', zorder=10)
        ax.scatter([basis_num], this_E[1] / np.sqrt(2), color='steelblue', marker='x', zorder=10)
        ax.scatter([basis_num], this_E[2] / np.sqrt(2), color='steelblue', marker='+', zorder=10)

 
    ax.set_ylim(0, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\sqrt{2}$')
    
    fig.savefig('different basis number.png', dpi=600)

     

