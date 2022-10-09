import numpy as np
import matplotlib.pyplot as plt
from q1_find_root import FindRootBis
plt.rcParams['axes.unicode_minus'] = False

# tanh x in (-1, 1), so solve for positive x
pts = 1000
res = np.zeros(pts)
t_range = np.linspace(0.01, 2, pts)

t_index = 0
# solve different equations for different t in t_range
for t in t_range:
    this_find = FindRootBis(lambda x: np.tanh(x / t) - x)
    # find the bracket! tanh(1) < 1, so try to find x_left s.t. tanh(x) > x
    x_left = 0
    while 1:
        if np.tanh(x_left / t) > x_left:
            this_find.find([1, x_left])
            res[t_index] = this_find.get_res()
            break
        x_left += 0.01
        # tanh x always smaller than x
        if x_left > 1:
            if t_index % 100 == 0:
                print(f'no solution for t = {t}')
            res[t_index] = 0.
            break
    t_index += 1
print('print part of the solution\n')
print('calculated m for t=\n')
x = [100 * i - 1 for i in range(10)]
print(t_range[x])
print('\n')
print('m=\n')
print(res[x])

fig, ax = plt.subplots(1, 1)
ax.plot(t_range, res, label='positive solutions', color='steelblue')
ax.plot(t_range, -res, label='negative solutions', color='steelblue', alpha=0.5)
ax.plot(t_range, 0*t_range, label='zero is always a solution', color='orange', alpha=0.5)
ax.set_xlabel('t')
ax.set_ylabel('m')
ax.set_title('m(t)')
ax.legend()
plt.show()
fig.savefig('q3.png')



