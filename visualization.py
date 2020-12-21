# Usage: python xxx.py path_to_pc1/pc2/output/epe3d/path_list [pc2]

import numpy as np
import sys
import mayavi.mlab as mlab
import os.path as osp
import pickle
import argparse

SCALE_FACTOR = 0.05
MODE = 'sphere'
DRAW_LINE = True


def get_correct_mask(sf_pred, sf_gt):
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1))

    return acc3d_relax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help="Path to directory containing points with npy of PCs")
    parser.add_argument('--relax', help='Display in Acc3DRelax format', action='store_true')

    args = parser.parse_args()
    visu_path = args.d
    show_relax = args.relax

    all_epe3d = np.load(osp.join(visu_path, 'epe3d_per_frame.npy'))

    path_list = None
    if osp.exists(osp.join(visu_path, 'sample_path_list.pickle')):
        with open(osp.join(visu_path, 'sample_path_list.pickle'), 'rb') as fd:
            path_list = pickle.load(fd)

    for index in range(len(path_list)):
        pc1 = np.load(osp.join(visu_path, 'pc1_' + str(index) + '.npy')).squeeze()
        pc2 = np.load(osp.join(visu_path, 'pc2_' + str(index) + '.npy')).squeeze()
        sf = np.load(osp.join(visu_path, 'sf_' + str(index) + '.npy')).squeeze()
        output = np.load(osp.join(visu_path, 'output_' + str(index) + '.npy')).squeeze()

        if pc1.shape[1] != 3:
            pc1 = pc1.T
            pc2 = pc2.T
            sf = sf.T
            output = output.T

        gt = pc1 + sf
        pred = pc1 + output

        if show_relax:
            correct = get_correct_mask(output, sf)
            pred = pred[correct]
            gt = gt[~correct]

        print('pc1, pc2, gt, pred', pc1.shape, pc2.shape, gt.shape, pred.shape)

        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), engine=None, size=(1600, 1000))

        if True:  # len(sys.argv) >= 4 and sys.argv[3] == 'pc1':
            mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0, 0, 1), scale_factor=SCALE_FACTOR, figure=fig,
                          mode=MODE)  # blue

        if len(sys.argv) >= 4 and sys.argv[3] == 'pc2':
            mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0, 1, 1), scale_factor=SCALE_FACTOR, figure=fig,
                          mode=MODE)  # cyan

        mlab.points3d(gt[:, 0], gt[:, 1], gt[:, 2], color=(1, 0, 0), scale_factor=SCALE_FACTOR, figure=fig,
                      mode=MODE)  # red
        mlab.points3d(pred[:, 0], pred[:, 1], pred[:, 2], color=(0, 1, 0), scale_factor=SCALE_FACTOR, figure=fig,
                      mode=MODE)  # green

        epe3d = all_epe3d[index]
        print(epe3d)
        path = path_list[index]
        print(path, epe3d)

        # DRAW LINE
        if DRAW_LINE and not show_relax:
            N = 2
            x = list()
            y = list()
            z = list()
            connections = list()

            inner_index = 0
            for i in range(gt.shape[0]):
                x.append(gt[i, 0])
                x.append(pred[i, 0])
                y.append(gt[i, 1])
                y.append(pred[i, 1])
                z.append(gt[i, 2])
                z.append(pred[i, 2])

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
        # DRAW LINE END

        mlab.view(90,  # azimuth
                  150,  # elevation
                  50,  # distance
                  [0, -1.4, 18],  # focalpoint
                  roll=0)

        mlab.orientation_axes()

        mlab.show()


if __name__ == '__main__':
    main()
