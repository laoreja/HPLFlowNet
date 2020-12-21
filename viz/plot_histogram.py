import argparse
import os.path as osp
import matplotlib.pyplot as plt

import numpy as np
plt.style.use('seaborn')

COLORS = {'blue': (57/255, 106/255, 177/255),
          'orange': (218/255, 124/255, 48/255),
          'red': (204/255, 37/255, 41/255),
          'green': (62/255, 150/255, 81/255)}

OUT_NAME = "histo.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help="Path to directory containing npy inputs")

    args = parser.parse_args()
    histo_path = args.d

    ours_hybrid = np.load(osp.join(histo_path, 'epe3d_histo1.npy'))[0]
    baseline = np.load(osp.join(histo_path, 'epe3d_histo2.npy'))[0]

    print("Mean ours hyb. = {}, # points = {}".format(np.mean(ours_hybrid), ours_hybrid.shape))
    print("Mean baseline = {}, # points = {}".format(np.mean(baseline), baseline.shape))

    plt.hist([baseline, ours_hybrid], bins=50, alpha=1.0, label=["HPLFlowNet", "Ours. Hybrid"])

    plt.legend()
    plt.yscale('log')
    plt.xlabel('EPE3D (m)')
    plt.ylabel('Frequency')
    plt.xlim(xmin=0)

    # plotting
    plt.savefig(osp.join(histo_path, OUT_NAME), bbox_inches='tight', dpi=1200, format='png')


if __name__ == '__main__':
    main()
