import numpy as np
import matplotlib.pyplot as plt

N = 20
# SI
e = 1.602176634e-19
me = 9.1093835e-31
hbar = 1.05457182e-34

Lw = 0.9
Lb = 0.1
U0 = 10


def v(x):
    if x < 0 or x >= Lw + Lb:
        x -= np.floor(x)

    if 0 <= x < Lw:
        return -U0 * Lb
    elif Lw <= x < Lb + Lw:
        return U0 * Lw
    assert 1, 'should not run here'


def get_vx(max_q_prime):
    '''
    NOTICE: p and q \in [-N, N], but q' \in [-2N, 2N].
    '''
    samp_x = np.linspace(0, 1, 2 * max_q_prime + 1)
    samp_v = np.array([])
    for x in samp_x:
        samp_v = np.append(samp_v, v(x))
    return samp_v


def kin_mat():
    '''
    hbar ^ 2 / me == 0.07619964434879783 eV * nm ** 2
    a = 1 nm
    '''
    k = 2 * np.pi ** 2 * 0.07619964434879783
    # only diagonal elements
    T = np.zeros((2 * N + 1, 2 * N + 1), dtype='complex')
    for i in range(2 * N + 1):
        q = i - N
        T[i, i] = k * q ** 2
    return T


def pot_mat(max_q_prime=2 * N):
    '''
    Hamiltonian matrix is 2N+1 by 2N+1.
    p-q \in [-2N, 2N].
    An example:

    [V00 V01 V02]
    [V10 V11 V12]
    [V20 V21 V22]
    In the above matrix, N == 1.

    V_{ij} corresponds to p == i - N and q == j - N
    so nonzero component of V is q' =  p - q = i - j

    NOTICE: p and q \in [-N, N], but q' \in [-2N, 2N].

    - try different ranges of q'

    - Are there eigenstates with lambda larger than a?
    '''
    assert type(max_q_prime) == int, 'range must be an int'
    V = np.zeros((2 * N + 1, 2 * N + 1), dtype='complex')
    vx = get_vx(max_q_prime)
    # want forward normalization to get V_{q'} compatible with psi
    v_q_primes = np.fft.fft(vx, norm='forward')
    for q_prime in range(-max_q_prime, max_q_prime + 1):
        # set components indexed with i - j == q_prime
        v_q_prime = v_q_primes[q_prime]
        for i in range(2 * N + 1):
            for j in range(2 * N + 1):
                if i - j == q_prime:
                    V[i, j] = v_q_prime
    return V


def calc_eigenvalues():
    global Lw, Lb
    es = None
    lws = np.linspace(0, 1, 1000)
    # lws = [0.9]
    for lw in lws:
        Lw = lw
        Lb = 1. - lw
        H = pot_mat() + kin_mat()

        # fig, ax = plt.subplots(1, 1)
        # s = ax.imshow(np.abs(H))
        # fig.colorbar(s)
        # fig.savefig('H.png', dpi=300)
        # plt.show()

        # fig, ax = plt.subplots(1, 1)
        # s = ax.imshow(np.abs(H - H.T.conj()))
        # fig.colorbar(s)
        # fig.savefig('H-H^H.png', dpi=300)

        # H is hermitian so energies will be real
        E, states = np.linalg.eig(H)
        lowest5 = np.sort(E.real)[:5]
        print('energies:', lowest5)
        if es is None:
            es = lowest5
        else:
            es = np.vstack((es, lowest5))

    # saw-like lines: not enough sampling does not notice slight change in Lw
    fig, ax = plt.subplots(1, 1)
    for i in range(es.shape[1]):
        e = es[:, i]
        ax.plot(lws, e, label=f'{i+1}-th lowest energy state')
    ax.legend()
    ax.set_xlabel('Lw / nm')
    plt.show()
    # fig.savefig('different_lws.png', dpi=300)


if __name__ == '__main__':
    calc_eigenvalues()
