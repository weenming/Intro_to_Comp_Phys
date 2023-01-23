import numpy as np
import matplotlib.pyplot as plt


def find_peak(f):
    peak_indices = np.array([], dtype=int)
    for i in range(1, f.shape[0] - 1):
        if f[i] > f[i + 1] > 0 and f[i] > f[i - 1]:
            peak_indices = np.append(peak_indices, i)
    return peak_indices


if __name__ == '__main__':
    t_space = np.loadtxt("sunspots.txt", dtype='double')[:, 1]
    print(t_space.shape)
    # zero peddling
    t_space = np.append(np.zeros(100000), t_space)
    t_space = np.append(t_space, np.zeros(100000))
    k_space = np.abs(np.fft.fft(t_space))
    k_space = np.fft.fftshift(k_space)**2
    k = np.fft.fftfreq(k_space.shape[0])
    k = np.fft.fftshift(k)

    peaks = []
    for i in find_peak(k_space):
        if k_space[i] > np.max(k_space) / 16:
            peaks.append(i)

    print([k[peak] for peak in peaks])
    peak = peaks[4]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(k[peak], k_space[peak],
               edgecolors='orange', c='none')
    ax.plot([k[peak], k[peak]], [0, np.max(k_space)],
            c='orange', alpha=0.3, ls='--', label=f'k = {k[peak]:.5f} month'+'$^{-1}$')
    ax.plot(k, k_space, label='FT of sun spots')
    ax.fill_betweenx([0, np.max(k_space)], 0, 1/np.loadtxt("sunspots.txt",
                     dtype='double')[:, 1].shape[0], color='grey', alpha=0.2, label='unreliable range')
    ax.legend()
    ax.set_xlim(0, 0.02)
    ax.set_xlabel('frequency / month'+'$^{-1}$')
    ax.set_ylabel('intensity of this frequency / a.u.')
    fig.savefig('q2_sunspots_zero_peddling.png', dpi=300)
    plt.show()
