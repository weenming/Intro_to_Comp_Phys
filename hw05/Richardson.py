import numpy as np
import matplotlib.pyplot as plt


def phi(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def D_n0(f, x, h, n):
    '''
    input:
        function f to take derivative
        variable x
        interpolate interval h
        max number of rows in Richardson table
    return:
        n by 1 nparray
    '''
    D = np.zeros((n, 1))
    for i in range(n):
        D[i, 0] = phi(f, x, h / (2 ** i))
    return D


def Richardson_table(D):
    n = D.shape[0]
    R = np.zeros((n, n))
    R[:, 0] = D[:, 0]
    for col in range(1, n):
        for row in range(col, n):
            R[row, col] = R[row, col - 1] + \
                (R[row, col - 1] - R[row - 1, col - 1]) / (4 ** col - 1)
    return R, R[n - 1, n - 1]


if __name__ == '__main__':
    n = 0
    res = 1e1000
    diff = []
    while (abs(res - 0.5) > 1e-10):
        n += 1
        R, res = Richardson_table(D_n0(np.sin, np.pi / 3, 1, n))
        diff.append(np.log10(abs(res - 0.5)))
    print('termination n is', n)
    print('result is ', res)
    print('theoretical value is', np.cos(np.pi / 3))
    print('Richardson table is\n', R)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(range(n), diff, label='error')
    ax.plot(ax.get_xlim(), [-6, -6], ls=':', color='grey')
    ax.set_ylabel('log 10 error')
    ax.set_xlabel('n')
    fig.show()
    fig.savefig('q1.png', dpi=400)
