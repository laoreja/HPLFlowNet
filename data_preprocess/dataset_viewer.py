#!/usr/bin/env python3
import argparse

import numpy as np
import mayavi.mlab as mlab
import os.path as osp
import os

MODE = 'sphere'
# red, green, blue
COLORS_LIST = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def is_quit(text):
    reply = str(input(text)).lower().strip()
    if reply[0] == 'q':
        return True
    if reply[0] == 'n':
        return False


def draw_line(pc1, pc2):
    N = 2
    x = list()
    y = list()
    z = list()
    connections = list()
    inner_index = 0
    for i in range(pc1.shape[0]):
        x.append(pc1[i, 0])
        x.append(pc2[i, 0])
        y.append(pc1[i, 1])
        y.append(pc2[i, 1])
        z.append(pc1[i, 2])
        z.append(pc2[i, 2])

        connections.append(np.vstack(
            [np.arange(inner_index, inner_index + N - 1.5),
             np.arange(inner_index + 1, inner_index + N - 0.5)]
        ).T)
        inner_index += N

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)

    connections = np.vstack(connections)

    src = mlab.pipeline.scalar_scatter(x, y, z)

    src.mlab_source.dataset.lines = connections
    src.update()

    lines = mlab.pipeline.tube(src, tube_radius=0.005, tube_sides=6)
    mlab.pipeline.surface(lines, line_width=2, opacity=.4, color=(1, 1, 0))


def plot_points(pc_sequence, scale_factor):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), engine=None, size=(1600, 1000))

    for idx, pc in enumerate(pc_sequence):
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=COLORS_LIST[idx],
                      scale_factor=scale_factor, figure=fig, mode=MODE)

    # draw a line between 2 PC
    pc1 = pc_sequence[0]
    pc1_sf = pc_sequence[1]
    draw_line(pc1, pc1_sf)

    # show all PCs
    mlab.view(90,  # azimuth
              150,  # elevation
              50,  # distance
              [0, -1.4, 18],  # focalpoint
              roll=0)
    mlab.orientation_axes()
    mlab.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help="Path to directory containing points with npy of PCs")
    parser.add_argument('-n', type=int, help='Number of points to sample from each PC', default=-1)
    parser.add_argument('-s', type=float, help='Scale factor for markers. Default is ', default=0.01)

    args = parser.parse_args()
    visu_path = args.d
    points_number = args.n
    scale_factor = args.s
    for sample_number in os.listdir(osp.join(visu_path)):
        # put all PC from time step t, t+1, t-1
        pc_sequence = []
        for pc_path in sorted(os.listdir(osp.join(visu_path, sample_number))):
            if pc_path.startswith("pc"):
                pc = np.load(osp.join(visu_path, sample_number, pc_path)).squeeze()
                # sample N points from a PC
                if points_number != -1:
                    indices = np.arange(pc.shape[0], dtype=np.long)
                    sampled_indices = np.random.choice(indices, size=points_number, replace=False, p=None)
                    pc = pc[sampled_indices]

                pc_sequence.append(pc)
                print(osp.join(visu_path, sample_number, pc_path), pc.shape)

        plot_points(pc_sequence, scale_factor)
        if is_quit("n - Show next, q - Quit: "):
            break


if __name__ == '__main__':
    main()
