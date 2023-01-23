import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False


f = lambda x, y: np.sin(x+y) + np.cos(x+2*y)

def plot_xyz(generation=None, current_res=None):
    x = np.linspace(0, 2*np.pi, 100)
    y = np.linspace(0, 2*np.pi, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_mesh, y_mesh, f(x_mesh, y_mesh))
    if generation is None:
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z = \sin(x+y) + \cos(x+2y)$')
        ax.set_zlim([-5, 5])
        plt.show()
        fig.savefig('q2_sketch.png')
    else:
        x = current_res['x']
        y = current_res['y']
        z = current_res['z']
        s = ax.scatter(x, y, z + 10, color='orange', alpha=1)
        s.set_edgecolors = s.set_facecolors = lambda *args: None
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set_zlim([-5, 5])
        ax.set_title(f'generation {generation}')
        ax.view_init(elev=90., azim=0)
        fig.savefig(f'q2_gen{generation}.png')
        plt.show()


# genetic algorithm
def mutation(herd):
    np.random.rand()
    herd += np.random.rand(herd_size, 2) * 0.1 - 0.05
    # restriction: may exceed allowed  range
    for i in range(herd_size):
        if 2 * np.pi > herd[i, 0] > 0 and 2 * np.pi > herd[i, 1] > 0:
            continue
        if herd[i, 0] < 0:
            herd[i, 0] = 0
        elif herd[i, 0] > 2 * np.pi:
            herd[i, 0] = 2 * np.pi
        if herd[i, 1] < 0:
            herd[i, 1] = 0
        elif herd[i, 1] > 2 * np.pi:
            herd[i, 1] = 2 * np.pi
    return herd


def cross(parent1, parent2):
    return (parent1 + parent2) / 2


def selection(parents):
    # no punish: selection and cross will not exceed the allowed range.
    merits = 1 / (f(parents[:, 0], parents[:, 1]) + 2.01)
    merits = merits / merits.sum()
    sons = np.zeros((herd_size, 2))
    for i in range(herd_size):
        rand1 = np.random.random()
        prob_sum1 = 0
        # select parent according to their merits: probability proprotional to merit
        for j in range(herd_size):
            prob_sum1 += merits[j]
            if prob_sum1 > rand1:
                parent1 = parents[j, :]
                break
        # select another parent
        rand2 = np.random.random()
        prob_sum2 = 0
        # select parent according to their merits: probability proprotional to merit
        for j in range(herd_size):
            prob_sum2 += merits[j]
            if prob_sum2 > rand2:
                parent2 = parents[j, :]
                break
        sons[i, :] = cross(parent1, parent2)
    return sons


herd_size = 1000
max_generation = 100

def main():
    # plot_xyz()
    # range {x, y} in {(0, 2pi), (0,2pi)}
    herd = np.random.random((herd_size, 2)) * 2 * np.pi
    plot_xyz(0, {'x': herd[:, 0], 'y': herd[:, 1], 'z': f(herd[:, 0], herd[:, 1])})
    print('begin optimization')
    for i in range(max_generation):
        print(f'generation {i}')
        avg_herd = herd.sum(0) / herd_size
        res = f(avg_herd[0], avg_herd[1])
        print(f'avg individual is {avg_herd}, avg fnuc {res}')
        herd = selection(herd)
        # herd = mutation(herd)
        if i % 1 == 0:
            pass
            # plot_xyz(i + 1, {'x': herd[:, 0], 'y': herd[:, 1], 'z': f(herd[:, 0], herd[:, 1])})

if __name__ == '__main__':
    main()
