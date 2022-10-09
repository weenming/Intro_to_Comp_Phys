import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['axes.unicode_minus'] = False


def plot_func(f):
    # sketch x^3-5x+3
    x = np.linspace(-5, 5, int(1e5))
    # f = lambda x: x ** 3 - 5 * x + 3
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, f(x), color='steelblue')
    ax.plot(x, 0 * x, color='orange', ls=':')
    ax.set_xlabel('x')

    ax.set_ylabel('function')
    ax.set_title('sketch')
    plt.show()
    fig.savefig('q1_sketch.png', dpi=600)


class FindRoot:
    def __init__(self, f, df=None):
        if df is None:
            self.df = self.set_df_num()
        else:
            self.df = df
        self.f = f
        self._res = None
        self._error = None

    def set_df_num(self):
        dx = 1e-5
        def df_num(x):
            return (self.f(x+dx) - self.f(x)) / dx
        return df_num

    def print_res(self):
        assert self._res is not None and self._error is not None, 'solution not found yet'
        print('solution is {:0.20f}, error is {:0.20f}'.format(self._res, self._error))

    def get_res(self):
        return self._res


class FindRootBis(FindRoot):
    def iter_bis(self, bracket):
        mid = (bracket[0] + bracket[1]) / 2
        if self.f(mid) < 0:
            bracket[1] = mid
        elif self.f(mid) > 0:
            bracket[0] = mid
        else:
            bracket[0] = mid
            bracket[1] = mid
            return bracket
        return bracket

    def find(self, bracket):
        mul = self.f(bracket[0]) * self.f(bracket[1])
        assert mul <= 0, \
            'invalid input bracket: f(bracket[0])*f(bracket[1]) must < 0'
        # if f(x) == 0, simply output this result.
        if mul == 0:
            if self.f(bracket[0]) == 0:
                bracket[1] = bracket[0]
            else:
                bracket[0] = bracket[1]
        # adjust so that f(bracket[0]) > 0
        elif self.f(bracket[0]) < 0:
            tmp = bracket[0]
            bracket[0] = bracket[1]
            bracket[1] = tmp

        # find root using binary search method
        while abs(bracket[0] - bracket[1]) > 1e-4:
            bracket = self.iter_bis(bracket)
        self._res = (bracket[0] + bracket[1]) / 2
        self._error = abs(bracket[0] - bracket[1]) / 2


class FindRootNewton(FindRoot):
    def iter_newton(self, x):
        df = self.df(x)
        assert df, 'error: derivative is zero!'
        return x - self.f(x) / df

    def find(self, x0):
        x_last = x0
        x = x0
        count = 0
        # polish up
        while count == 0 or abs(x_last - x) > 1e-14:
            x_last = x
            x = self.iter_newton(x_last)
            if x == x_last:
                print('warning: f(x) may underflow, error is imprecise!')
                pass
            count += 1
            if count > 100000:
                assert False, 'too many iterations in Newton method'
        self._res = x
        self._error = abs(x_last - x)


class FindRootHybrid(FindRootNewton, FindRootBis):
    def find(self, bracket):
        mul = self.f(bracket[0]) * self.f(bracket[1])
        assert mul <= 0, \
            'invalid input bracket: f(bracket[0])*f(bracket[1]) must < 0'
        # if f(x) == 0, simply output this result.
        if mul == 0:
            if self.f(bracket[0]) == 0:
                bracket[1] = bracket[0]
            else:
                bracket[0] = bracket[1]
        # adjust so that f(bracket[0]) > 0
        elif self.f(bracket[0]) < 0:
            tmp = bracket[0]
            bracket[0] = bracket[1]
            bracket[1] = tmp

        epsilon = 1e-14
        while abs(bracket[0] - bracket[1]) > epsilon:
            mid = (bracket[0] + bracket[1]) / 2
            # if derivative none zero and new x in range, use Newton
            if self.df(mid) != 0:
                newx = mid - self.f(mid) / self.df(mid)
                if (bracket[0] - newx) * (bracket[1] - newx) < 0:
                    # update using Newton
                    newf = self.f(newx)
                    if newf < 0:
                        bracket[1] = newx
                        continue
                    elif newf > 0:
                        bracket[0] = newx
                        continue
                    # else:
                    #     print('warning: f(x) may underflow, error is imprecise!')
                        # bracket[0] = newx
                        # bracket[1] = newx

            # this should happen rarely so a redundant adding is acceptable
            bracket = self.iter_bis(bracket)

        self._res = (bracket[0] + bracket[1]) / 2
        self._error = abs(bracket[0] - bracket[1]) / 2


f = lambda x: x ** 3 - 5 * x + 3
df = lambda x: 3 * x**2 -5
def find_fast(low, high):
    # mul = f(bracket[0]) * f(bracket[1])
    # assert mul <= 0, \
    #     'invalid input bracket: f(bracket[0])*f(bracket[1]) must < 0'
    # # if f(x) == 0, simply output this result.
    # if mul == 0:
    #     if f(bracket[0]) == 0:
    #         bracket[1] = bracket[0]
    #     else:
    #         bracket[0] = bracket[1]
    # # adjust so that f(bracket[0]) > 0
    # elif f(bracket[0]) < 0:
    #     tmp = bracket[0]
    #     bracket[0] = bracket[1]
    #     bracket[1] = tmp

    last_is_not_newton = True
    while high - low > 1e-14:
        mid = (low + high) / 2
        # if derivative none zero and new x in range, use Newton
        if last_is_not_newton and df(mid) != 0:
            newx = mid - f(mid) / df(mid)
            if high > newx > low:
                # update using Newton
                newf = f(newx)
                lowf = f(low)
                if newf * lowf > 0:
                    low = newx
                else:
                    high = newx
                last_is_not_newton = False
                continue
        # this should happen rarely so a redundant adding is acceptable
        last_is_not_newton = True
        lowf = f(low)
        midf = f(mid)
        if midf > 0 and lowf > 0 or midf < 0 and lowf < 0:
            low = mid
        else:
            high = mid


def measure_time():
    # equation to solve
    f = lambda x: x ** 3 - 5 * x + 3

    # sketch the function
    # plot_func(f)
    begin = time.time()
    for i in range(1000):
        # bisection method
        # print('bisection method, solution 1')
        find1 = FindRootBis(f)
        find1.find([0, 1])
        # find1.print_res()

        # print('bisection method, solution 2')
        find2 = FindRootBis(f)
        find2.find([1, 2])
        # find2.print_res()
    end = time.time()
    print(end-begin)
    res1 = find1.get_res()
    res2 = find2.get_res()

    begin = time.time()
    for i in range(1000):
        # 'polish up' using Newton method
        # provide analytic derivative for better time complexity
        # print('Newton method, solution 1')
        find3 = FindRootNewton(f, lambda x: 3 * x**2 - 5)
        # not providing the analytical derivative also works
        # find3 = FindRootNewton(f)
        # waring: Newton method converges fast. When f(x) converges faster than x, unintended underflow may occur!
        find3.find(res1)
        # find3.print_res()

        # print('Newton method, solution 2')
        find4 = FindRootNewton(f, lambda x: 3 * x**2 - 5)
        find4.find(res2)
        # find4.print_res()
    end = time.time()
    print(end - begin)

    begin = time.time()
    for i in range(1000):
        # find using hybrid method
        # provide analytic derivative for better time complexity
        # print('Hybrid method, solution 1')
        # find5 = FindRootHybrid(f, lambda x: 3 * x**2 - 5)
        find_fast(0, 1.6)
        # find5.print_res()

        # print('Hybrid method, solution 2')
        # find6 = FindRootHybrid(f, lambda x: 3 * x**2 - 5)
        find_fast(1.6, 2)
        # find6.print_res()
    end = time.time()
    print(end-begin)


def main():
    # equation to solve
    f = lambda x: x ** 3 - 5 * x + 3

    # sketch the function
    plot_func(f)

    # bisection method
    print('bisection method, solution 1')
    find1 = FindRootBis(f)
    find1.find([0, 1])
    find1.print_res()

    print('bisection method, solution 2')
    find2 = FindRootBis(f)
    find2.find([1, 2])
    find2.print_res()
    res1 = find1.get_res()
    res2 = find2.get_res()

    # 'polish up' using Newton method
    # provide analytic derivative for better time complexity
    print('Newton method, solution 1')
    find3 = FindRootNewton(f, lambda x: 3 * x ** 2 - 5)
    # not providing the analytical derivative also works
    # find3 = FindRootNewton(f)
    # waring: Newton method converges fast. When f(x) converges faster than x, unintended underflow may occur!
    find3.find(res1)
    find3.print_res()

    print('Newton method, solution 2')
    find4 = FindRootNewton(f, lambda x: 3 * x ** 2 - 5)
    find4.find(res2)
    find4.print_res()

    # find using hybrid method
    # provide analytic derivative for better time complexity
    print('Hybrid method, solution 1')
    find5 = FindRootHybrid(f, lambda x: 3 * x**2 - 5)
    find5.find([0, 1])
    find5.print_res()

    print('Hybrid method, solution 2')
    find6 = FindRootHybrid(f, lambda x: 3 * x**2 - 5)
    find6.find([1, 2])
    find6.print_res()

if __name__ == '__main__':
    main()
