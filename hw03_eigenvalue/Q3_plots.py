import Q3_schrodinger as q3
import matplotlib.pyplot as plt
import numpy as np

def plot_basis_center_func():
    centers = [i for i in range(-5, 5)]
    basis = [q3.basis_pos(center, odd=False, w=3) for center in centers]
    q3.plot_wave_func(basis, [basis[i](0) for i in range(len(basis))], c=lambda x: plt.get_cmap('viridis')(x/5), label=0)

def plot_basis_width_func():
    ws = [i for i in np.linspace(0.1, 1, 6)]
    basis = [q3.basis_wid(w, odd=False) for w in ws]
    q3.plot_wave_func(basis, [basis[i](0) for i in range(len(basis))], c=lambda x: plt.get_cmap('twilight')(x/5), label=0)

def plot_basis_width_func_odd_and_even():
    ws = [i for i in np.linspace(0.1, 1, 3)]
    basis = [q3.basis_wid(w, odd=False) for w in ws] + [q3.basis_wid(w, odd=True) for w in ws]
    q3.plot_wave_func(basis, [basis[i](0) for i in range(len(basis))], c=lambda x: plt.get_cmap('twilight')(x/5), label=0)

def plot_E_to_basis_num_V2_width():
    fpath = './result/V2/different_width_odd_and_even/Es.txt'
    Es = np.loadtxt(fpath)
    basis_nums = [3, 4, 5, 6, 7, 8, 9, 10]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    fig.set_size_inches(10, 6)
    for y in Es[-1, :]:
        ax.plot([2, 11], [y / np.sqrt(2), y / np.sqrt(2)], ls='-', color='grey', alpha=0.3, zorder=0)
        print(y)
    label = lambda x: f'energy of {x % 2 + 1}-th even wave function' if x < 2 else f'energy of {x % 2 + 1}-th odd wave function'
    markerls = ['+', 'x', '+', 'x']
    colorls = ['orange', 'orange', 'steelblue', 'steelblue']
    for i in range(Es.shape[1]):
        ax.scatter(basis_nums, Es[:, i] / np.sqrt(2), color = colorls[i], marker = markerls[i], zorder=1, label=label(i))
    ax.set_ylim(-np.sqrt(2) / 4, None)
    ax.set_xlabel('basis number')
    ax.legend()
    # V2 = (x^4/x_0^4 - x^2/x_0^2) \frac{m\omega^2x_0^2}{2} where x_0 = 1
    ax.set_ylabel('energy / '+'$\hbar\omega$')
    fig.savefig('./result/V2/different_width_odd_and_even/different_basis.png', dpi=600)
    plt.show()


def plot_E_to_basis_num_V2_center():
    fpath = './result/V2/different_center/Es.txt'
    Es = np.loadtxt(fpath)
    basis_nums = [4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 35, 40, 100]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)
    fig.set_size_inches(10, 6)
    for y in Es[-1, :]:
        ax.plot([3, 41], [y / np.sqrt(2), y / np.sqrt(2)], ls='-', color='grey', alpha=0.3, zorder=0)
        print(y)

    markerls = ['+', 'x', '.', 'v']
    colors = ['orange'] * 4
    for i in range(Es.shape[1]):
        ax.scatter(basis_nums, Es[:, i] / np.sqrt(2), color = colors[i], marker = markerls[i], zorder=1, label=f'energy of {i}-th wave function')
    ax.set_ylim(-np.sqrt(2) / 4, None)
    ax.set_xlabel('basis number')
    ax.legend(loc='upper right')
    # V2 = (x^4/x_0^4 - x^2/x_0^2) \frac{m\omega^2x_0^2}{2} where x_0 = 1
    ax.set_ylabel('energy / '+'$\hbar\omega$')

    ax_inset = ax.inset_axes([0.44, 0.2, 0.5, 0.5])
    for y in Es[-1, :]:
        ax_inset.plot([19, 41], [y / np.sqrt(2), y / np.sqrt(2)], ls='-', color='grey', alpha=0.3, zorder=0)
        print(y)

    markerls = ['+', 'x', '.', 'v']
    colors = ['orange'] * 4
    for i in range(Es.shape[1]):
        ax_inset.scatter(basis_nums[7:], Es[7:, i] / np.sqrt(2), color = colors[i], marker = markerls[i], zorder=1, label=f'energy of {i}-th wave function')
    ax_inset.set_ylim(-np.sqrt(2) / 4, None)
    ax_inset.set_xlabel('basis number')
    # V2 = (x^4/x_0^4 - x^2/x_0^2) \frac{m\omega^2x_0^2}{2} where x_0 = 1
    ax_inset.set_ylabel('energy / '+'$\hbar\omega$')
    
    fig.savefig('./result/V2/different_center/different_basis.png', dpi=600)
    plt.show()

plot_basis_width_func()