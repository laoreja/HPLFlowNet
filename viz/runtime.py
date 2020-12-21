import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')


def plot_runtime(x, y, out_file):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=r'$K_i$', ylabel='Inference time (ms)')

    plt.xlim((0, 30))
    plt.ylim(ymax=8000)

    plt.savefig(out_file, bbox_inches='tight', dpi=1200, format='png')


# FlyingThings3D
x = np.array([1, 5, 10, 20, 30])
y = np.array([542.8, 1593.1, 2715, 5231, 7853.8])
plot_runtime(x, y, "fly_runtime.png")

# KITTI
x = np.array([1, 5, 10, 20, 30])
y = np.array([485.8, 1380.9, 2474.0, 4700.8, 6930.0])
plot_runtime(x, y, "kitti_runtime.png")
