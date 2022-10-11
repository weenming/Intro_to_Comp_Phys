from locale import normalize
from os import stat
from re import S
from typing import Iterator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# if these constants are changed, params in psi need to change, as well
hbar = 1.
m = 1.

def V1(x):
    # harmonic oscillator, omega = \sqrt{2}
    return x ** 2

def V2(x):
    return x ** 4 - (x) ** 2 


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

def inner_product(func1, func2, lower=-50, upper=50, step=1e-2):
    x = np.linspace(lower, upper, int((upper - lower) / step))
    return np.dot(func1(x), func2(x)) * step

def calc_hamilton(basis, v=V1):
    '''
    basis: python list of wave functions
    '''
    print('start calculating H')
    l = len(basis)
    hamilton = np.zeros((l, l))
    for i in tqdm(range(l)):
        for j in tqdm(range(l), leave=False):
            hamilton[i, j] = inner_product(
                basis[i], hamilton_operator(basis[j], v))
            
    return hamilton

def calc_s(basis):
    print('start calculating s')
    l = len(basis)
    s = np.zeros((l, l))
    for i in tqdm(range(l)):
        for j in tqdm(range(l), leave=False):
            s[i, j] = inner_product(basis[i], basis[j])
    return s


def basis_pos(center, w=1, odd=False):
    if odd:
        return lambda x: x * (2 * w / np.pi) ** 0.25 * np.exp(- w * (x - center) ** 2)

    return lambda x: (2 * w / np.pi) ** 0.25 * np.exp(- w * (x - center) ** 2)

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
    assert min(D) > 0, 'S must be positive definite'
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

def plot_wave_func(states, E, fpath_and_fname=f'tmp', plot_abs=False, c=lambda x: 'orange', label=None):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 9)
    x = np.linspace(-10, 10, 1000)
    for i in range(len(states)):
        if label is None:
            l = f'{i+1}-th wave function with energy {E[i] / np.sqrt(2)}'+'$\cdot\sqrt{2}$'
            order = i
        elif label == 0:
            l=None
            order=i
        else:
            l = f'{label(i)}-th wave function with energy {E[i] / np.sqrt(2)}'+'$\cdot\sqrt{2}$'
            order = label(i)
        if plot_abs:
            ax.plot(x, np.power(states[i](x), 2), label=l ,color=c(i), zorder=order)
            ax.set_ylabel('$|\psi|$')
        else:
            ax.plot(x, states[i](x), label=l ,color=c(i), zorder=order)
            ax.set_ylabel('$\psi$')
    ax.set_xlabel('x')
    
    ax.legend()
    # ax.set_title(f'total {len(states)} basis')
    fig.savefig(fpath_and_fname, dpi=600)
    # plt.show()


def main_width_dif_basis_num(basis_num, is_odd, v=V1):
    # 这样就不行，算出来的全是同一个。原因是lazy evaluation，
    # 在调用basis[x]的时候回到原来scope里，psi里的w全都是for循环结束前的最后一个
    # basis = basis_wids(np.linspace(0.1, 1, 10))
    # 这样就可以通过多次调用函数创造不同的scope
    #larger param w, wider basis wave function
    basis = [basis_wid(w, is_odd) for w in np.linspace(0.1, 1, basis_num)]
    H = calc_hamilton(basis, v=v)
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

    return E, eigen_states_normalized

def main_width_dif_basis_num_even_and_odd(basis_num, v=V1):
    basis = [basis_wid(w, odd=False) for w in np.linspace(0.1, 1, basis_num)]
    basis += [basis_wid(w, odd=True) for w in np.linspace(0.1, 1, basis_num)]
    H = calc_hamilton(basis, v=v)
    S = calc_s(basis)
    # eigenvalue problem: H'C' = EC', where H' = S^{-1/2}HS^{1/2}, C' = S^{1/2}C
    _, S_sqrt_neg = matrix_sqrt(S)
    H_primed = S_sqrt_neg.dot(H).dot(S_sqrt_neg)
    E, C_primed = np.linalg.eig(H_primed)
    print(E)
    C = np.dot(S_sqrt_neg, C_primed)
    C_normalized = normalize(C, S)
    eigen_states_normalized = [get_eigenstates(C_normalized, basis, j) for j in range(len(basis))]   
    # sort energy and states by E (sorted in ascending order)
    eigen_states_normalized = [x for _, x in sorted(zip(E, eigen_states_normalized), reverse=False)]
    E.sort()
    # plot and save figure
    # plot_wave_func(eigen_states_normalized, E)
    print('check normalization:')
    print([inner_product(eigen_states_normalized[i], eigen_states_normalized[i]) for i in range(len(basis))])

    return E, eigen_states_normalized

def main_center_dif_basis_num_even_and_odd(basis_num, v=V1):
    basis = [basis_pos(center, odd=False) for center in np.linspace(-5, 5, basis_num)]
    basis += [basis_pos(center, odd=True) for center in np.linspace(-5, 5, basis_num)]
    H = calc_hamilton(basis, v=v)
    S = calc_s(basis)
    # eigenvalue problem: H'C' = EC', where H' = S^{-1/2}HS^{1/2}, C' = S^{1/2}C
    _, S_sqrt_neg = matrix_sqrt(S)
    H_primed = S_sqrt_neg.dot(H).dot(S_sqrt_neg)
    E, C_primed = np.linalg.eig(H_primed)
    print(E)
    C = np.dot(S_sqrt_neg, C_primed)
    C_normalized = normalize(C, S)
    eigen_states_normalized = [get_eigenstates(C_normalized, basis, j) for j in range(len(basis))]   
    # sort energy and states by E (sorted in ascending order)
    eigen_states_normalized = [x for _, x in sorted(zip(E, eigen_states_normalized), reverse=False)]
    E.sort()
    # plot and save figure
    # plot_wave_func(eigen_states_normalized, E)
    print('check normalization:')
    print([inner_product(eigen_states_normalized[i], eigen_states_normalized[i]) for i in range(len(basis))])

    return E, eigen_states_normalized

def main_center_dif_basis_num_even(centers, v=V1, w=1):
    basis = [basis_pos(center, odd=False, w=w) for center in centers]
    H = calc_hamilton(basis, v=v)
    S = calc_s(basis)
    # eigenvalue problem: H'C' = EC', where H' = S^{-1/2}HS^{1/2}, C' = S^{1/2}C
    _, S_sqrt_neg = matrix_sqrt(S)
    H_primed = S_sqrt_neg.dot(H).dot(S_sqrt_neg)
    E, C_primed = np.linalg.eig(H_primed)
    print(E)
    C = np.dot(S_sqrt_neg, C_primed)
    C_normalized = normalize(C, S)
    eigen_states_normalized = [get_eigenstates(C_normalized, basis, j) for j in range(len(basis))]   
    # sort energy and states by E (sorted in ascending order)
    eigen_states_normalized = [x for _, x in sorted(zip(E, eigen_states_normalized), reverse=False)]
    E.sort()
    # plot and save figure
    # plot_wave_func(eigen_states_normalized, E)
    print('check normalization:')
    print([inner_product(eigen_states_normalized[i], eigen_states_normalized[i]) for i in range(len(basis))])

    return E, eigen_states_normalized



def different_basis_width_pot_1(fpath = './result/V1/different_width_odd_and_even/'):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    fig_wavefunc, ax_wavefunc = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([2, 11], [y, y], ls='-', color='grey', alpha=0.3, zorder=0)
    cmap = plt.get_cmap('viridis')
    for basis_num in [3, 4, 5, 6, 7, 8, 9, 10]:
        this_E_even, this_eigenstates_even = main_width_dif_basis_num(basis_num, is_odd=False)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E_even[0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[0] / np.sqrt(2) - 0.5)) + 14) / 20), marker='.', zorder=10)
        ax.scatter([basis_num], this_E_even[1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[1] / np.sqrt(2) - 2.5)) + 14) / 20), marker='x', zorder=10)
        ax.scatter([basis_num], this_E_even[2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[2] / np.sqrt(2) - 4.5)) + 14) / 20), marker='+', zorder=10)
        np.savetxt(fpath+f'basis_num{basis_num}_even.txt', this_E_even)
        
        this_E_odd, this_eigenstates_odd = main_width_dif_basis_num(basis_num, is_odd=True)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E_odd[0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[0] / np.sqrt(2) - 1.5)) + 14) / 20), marker='.', zorder=10)
        ax.scatter([basis_num], this_E_odd[1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[1] / np.sqrt(2) - 3.5)) + 14) / 20), marker='x', zorder=10)
        ax.scatter([basis_num], this_E_odd[2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[2] / np.sqrt(2) - 5.5)) + 14) / 20), marker='+', zorder=10)
        np.savetxt(fpath+f'basis_num{basis_num}_odd.txt', this_E_odd)

        plot_wave_func(this_eigenstates_even[:2]+this_eigenstates_odd[:2], np.append(this_E_even[:2], this_E_odd[:2]), fpath+f'wavefuncplot_basis_num={basis_num}.png', plot_abs=False, c=lambda x: plt.get_cmap('Oranges')(- x * 60 + 150) if x < 2 else plt.get_cmap('Blues')(- x * 60 + 270), label=lambda x: [1, 3, 2, 4][x])
    # add color bar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical')
    cb1.set_ticklabels([8.3e-7, 4.5e-5, 2.5e-3, 13.e-1, 7.3, 70])
    cb1.set_label('$\ln(|E_i-E_i^{theory}|/\hbar\omega)$')
    fig.add_axes(ax_cb)

    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={0.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={2.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={4.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={1.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={3.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={5.5}\hbar\omega$')
    
    ax.set_ylim(0, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    
    fig.savefig(fpath+'different basis number.png', dpi=600)

def different_basis_width_pot_1_only_even(fpath = './result/V1/different_width_even/'):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    fig_wavefunc, ax_wavefunc = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([2, 11], [y, y], ls='-', color='grey', alpha=0.3, zorder=0)
    cmap = plt.get_cmap('viridis')
    
    for basis_num in [3, 4, 5, 6, 7, 8, 9, 10]:
        this_E_even, this_eigenstates_even = main_width_dif_basis_num(basis_num, is_odd=False)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E_even[0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[0] / np.sqrt(2) - 0.5)) + 14) / 20), marker='.', zorder=10)
        ax.scatter([basis_num], this_E_even[1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[1] / np.sqrt(2) - 2.5)) + 14) / 20), marker='x', zorder=10)
        ax.scatter([basis_num], this_E_even[2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[2] / np.sqrt(2) - 4.5)) + 14) / 20), marker='+', zorder=10)
        np.savetxt(fpath+f'basis_num{basis_num}_even.txt', this_E_even)

        plot_wave_func(this_eigenstates_even[:3], this_E_even[:3], fpath+f'wavefuncplot_basis_num={basis_num}.png', plot_abs=False, c=lambda x: plt.get_cmap('Oranges')(- x * 40 + 150), label=lambda x: x+1)
    # add color bar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical')
    cb1.set_ticklabels([8.3e-7, 4.5e-5, 2.5e-3, 13.e-1, 7.3, 70])
    cb1.set_label('$\ln(|E_i-E_i^{theory}|/\hbar\omega)$')
    fig.add_axes(ax_cb)

    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={0.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={2.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={4.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={1.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={3.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={5.5}\hbar\omega$')
    
    ax.set_ylim(0, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    
    fig.savefig(fpath+'different basis number.png', dpi=600)

def different_basis_center_pot_1(fpath = './result/V1/different_center_even/'):
    # basis_nums = [40]
    # for basis_num in basis_nums:
    #     this_E, eigenstates = main_center_dif_basis_num_even(basis_num, v=V1)
    #     plot_wave_func(eigenstates, this_E)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    fig_wavefunc, ax_wavefunc = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([3, 41], [y, y], ls='-', alpha=0.3, color='grey', zorder=0)
    basis_nums = [4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    this_E_evens = np.zeros((len(basis_nums), 4))
    i = 0
    for basis_num in basis_nums:
        this_E_even, this_eigenstates_even = main_center_dif_basis_num_even(np.linspace(-10, 10, basis_num), v=V1, w=3)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        cmap = plt.get_cmap('viridis') # this cmap normalizes to 1 (input \in (0, 1))
        plot_wave_func(this_eigenstates_even[:4], this_E_even[:4], fpath+f'wavefuncplot_basis_num={basis_num}.png', c=lambda x: plt.get_cmap('Oranges')(- x * 40 + 150))
        this_E_evens[i, :] = this_E_even[:4]
        i += 1
        np.savetxt(fpath+f'basis_num{basis_num}_even.txt', this_E_even)
    print(this_E_evens)

    ax.scatter(basis_nums, this_E_evens[:, 0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 0] / np.sqrt(2) - 0.5)) + 14) / 20), marker='.', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 1] / np.sqrt(2) - 1.5)) + 14) / 20), marker='x', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 2] / np.sqrt(2) - 2.5)) + 14) / 20), marker='+', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 3] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 3] / np.sqrt(2) - 3.5)) + 14) / 20), marker='v', zorder=10)
    
    # add color bar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical')
    cb1.set_ticklabels([8.3e-7, 4.5e-5, 2.5e-3, 13.e-1, 7.3, 70])
    cb1.set_label('$\ln(|E_i-E_i^{theory}|/\hbar\omega)$')
    fig.add_axes(ax_cb)

    ax.scatter([],[], color=cmap(0.5), marker='.', label=f'$E={0.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='x', label=f'$E={1.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='+', label=f'$E={2.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='v', label=f'$E={3.5}\hbar\omega$')

    ax.set_ylim(-np.sqrt(2) / 4, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    

    
    fig.add_axes(ax_cb)
    
    fig.savefig(fpath+'different basis number.png', dpi=600)
   
def different_basis_width_pot_2(fpath = './result/V2/different_width_odd_and_even/'):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    fig_wavefunc, ax_wavefunc = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([2, 11], [y, y], ls='-', color='grey', alpha=0.3, zorder=0)
    cmap = plt.get_cmap('viridis')
    basis_nums = [3, 4, 5, 6, 7, 8, 9, 10]
    this_Es = np.zeros((len(basis_nums), 4))
    i = 0
    for basis_num in basis_nums:
        this_E_even, this_eigenstates_even = main_width_dif_basis_num(basis_num, is_odd=False, v=V2)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E_even[0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[0] / np.sqrt(2) - 0.5)) + 14) / 20), marker='.', zorder=10)
        ax.scatter([basis_num], this_E_even[1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[1] / np.sqrt(2) - 2.5)) + 14) / 20), marker='x', zorder=10)
        ax.scatter([basis_num], this_E_even[2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_even[2] / np.sqrt(2) - 4.5)) + 14) / 20), marker='+', zorder=10)
        np.savetxt(fpath+f'basis_num{basis_num}_even.txt', this_E_even)
        
        this_E_odd, this_eigenstates_odd = main_width_dif_basis_num(basis_num, is_odd=True, v=V2)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        ax.scatter([basis_num], this_E_odd[0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[0] / np.sqrt(2) - 1.5)) + 14) / 20), marker='.', zorder=10)
        ax.scatter([basis_num], this_E_odd[1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[1] / np.sqrt(2) - 3.5)) + 14) / 20), marker='x', zorder=10)
        ax.scatter([basis_num], this_E_odd[2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_odd[2] / np.sqrt(2) - 5.5)) + 14) / 20), marker='+', zorder=10)
        np.savetxt(fpath+f'basis_num{basis_num}_odd.txt', this_E_odd)
        this_Es[i, :] = np.append(this_E_even[:2], this_E_odd[:2])
        i += 1

        plot_wave_func(this_eigenstates_even[:2]+this_eigenstates_odd[:2], np.append(this_E_even[:2], this_E_odd[:2]), fpath+f'wavefuncplot_basis_num={basis_num}.png', plot_abs=False, c=lambda x: plt.get_cmap('Oranges')(- x * 60 + 150) if x < 2 else plt.get_cmap('Blues')(- x * 60 + 270), label=lambda x: [1, 3, 2, 4][x])
    print(this_Es)
    np.savetxt(fpath+'Es.txt', this_Es)

    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={0.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={2.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={4.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='.', label=f'$E={1.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='x', label=f'$E={3.5}\hbar\omega$')
    ax.scatter([],[],color=cmap(0), marker='+', label=f'$E={5.5}\hbar\omega$')
    
    ax.set_ylim(0, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    
    fig.savefig(fpath+'different basis number.png', dpi=600)

def different_basis_center_pot_2(fpath = './result/V2/different_center/'):
        # basis_nums = [40]
    # for basis_num in basis_nums:
    #     this_E, eigenstates = main_center_dif_basis_num_even(basis_num, v=V1)
    #     plot_wave_func(eigenstates, this_E)
    
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    fig_wavefunc, ax_wavefunc = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    for y in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ax.plot([3, 41], [y, y], ls='-', alpha=0.3, color='grey', zorder=0)
    basis_nums = [4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 35, 40, 100]
    this_E_evens = np.zeros((len(basis_nums), 4))
    i = 0
    for basis_num in basis_nums:
        this_E_even, this_eigenstates_even = main_center_dif_basis_num_even(np.linspace(-10, 10, basis_num), v=V2, w=3)
        # np.savetxt(f'energy_basis_numeber{basis_num}.txt', this_E)
        cmap = plt.get_cmap('viridis') # this cmap normalizes to 1 (input \in (0, 1))
        plot_wave_func(this_eigenstates_even[:4], this_E_even[:4], fpath+f'wavefuncplot_basis_num={basis_num}.png', c=lambda x: plt.get_cmap('Oranges')(- x * 40 + 150))
        this_E_evens[i, :] = this_E_even[:4]
        i += 1
        np.savetxt(fpath+f'basis_num{basis_num}_even.txt', this_E_even)
    print(this_E_evens)
    np.savetxt(fpath+'Es.txt', this_E_evens)

    ax.scatter(basis_nums, this_E_evens[:, 0] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 0] / np.sqrt(2) - 0.5)) + 14) / 20), marker='.', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 1] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 1] / np.sqrt(2) - 1.5)) + 14) / 20), marker='x', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 2] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 2] / np.sqrt(2) - 2.5)) + 14) / 20), marker='+', zorder=10)
    ax.scatter(basis_nums, this_E_evens[:, 3] / np.sqrt(2), color=cmap((np.log(np.abs(this_E_evens[:, 3] / np.sqrt(2) - 3.5)) + 14) / 20), marker='v', zorder=10)
    
    # add color bar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical')
    cb1.set_ticklabels([8.3e-7, 4.5e-5, 2.5e-3, 13.e-1, 7.3, 70])
    cb1.set_label('$\ln(|E_i-E_i^{theory}|/\hbar\omega)$')
    fig.add_axes(ax_cb)

    ax.scatter([],[], color=cmap(0.5), marker='.', label=f'$E={0.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='x', label=f'$E={1.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='+', label=f'$E={2.5}\hbar\omega$')
    ax.scatter([],[], color=cmap(0.5), marker='v', label=f'$E={3.5}\hbar\omega$')

    ax.set_ylim(-np.sqrt(2) / 4, None)
    ax.set_xlabel('basis number')
    ax.legend()
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    

    
    fig.add_axes(ax_cb)
    
    fig.savefig(fpath+'different basis number.png', dpi=600)

if __name__ == '__main__':
    different_basis_center_pot_2()
