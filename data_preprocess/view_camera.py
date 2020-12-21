import math
import pickle
import sys
import os
import os.path as osp
import argparse
import mayavi.mlab as mlab

import IO
import numpy as np

import dataset_viewer as viewer

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, help="path to the raw data")

args = parser.parse_args()
root_path = args.raw_data_path


scene_list = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
keyframe_list = ['keyframe_1', 'keyframe_2', 'keyframe_5']


def plot_camera_sequence(C_list):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), engine=None, size=(1600, 1000))
    mlab.points3d(C_list[0][0], C_list[0][1], C_list[0][2], color=(1, 0, 0), scale_factor=0.008, figure=fig, mode='cube')
    for i in range(1, len(C_list)):
        # camera center
        center = C_list[i]
        mlab.points3d(center[0], center[1], center[2], color=(0, 1, 0), scale_factor=0.005, figure=fig, mode='sphere')

    # show all PCs
    mlab.view(90,  # azimuth
              150,  # elevation
              50,  # distance
              [0, -1.4, 18],  # focalpoint
              roll=0)
    mlab.orientation_axes()
    mlab.show()


def rotation_to_euler(R):
    """

    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).

    Args:
        R: rotation matrix

    Returns:
        array of euler angles
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def plot_frustum(Rt_list, img_list, f):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), engine=None, size=(1600, 1000))
    for i in range(len(Rt_list)):
        R = Rt_list[i][:3, :3]
        t = Rt_list[i][:3, 3]
        C = -np.dot(R.T, t)
        pp = np.array([0, 0, f])
        img = img_list[i][:, :, 1]
        euler = rotation_to_euler(R[:3, :3])
        obj = mlab.imshow(img.T, figure=fig)
        obj.actor.orientation = np.rad2deg(euler)
        obj.actor.position = (np.dot(R, pp) + t) + C
        # obj.actor.scale = [0.01, 0.01, 0.01]
        # center
    mlab.orientation_axes()
    mlab.show()


if __name__ == '__main__':
    param_list = []
    for scene in scene_list:
        for keyframe in keyframe_list:
            seq_path = osp.join(scene, keyframe)
            interval_root = osp.join(root_path, seq_path)
            intervals_list = os.listdir(interval_root)
            # Remove duplicated intervals
            interval_dict = {}
            for interval in intervals_list:
                sequence_beg = interval.split('_')[0]
                sequence_end = interval.split('_')[1]
                if sequence_beg in interval_dict and int(sequence_end) < int(interval_dict[sequence_beg]):
                    continue
                interval_dict[sequence_beg] = sequence_end

            intervals_list = [k + '_' + v for k, v in interval_dict.items()]
            intervals_list.sort()
            # loop over sorted intervals
            # Visualize the frustum for each interval
            for interval in intervals_list:
                interval_path = osp.join(interval_root, interval)
                interval_file_list = [f for f in os.listdir(osp.join(interval_path, "depth")) if f.endswith('.png')]
                name_list = list(map(lambda f: f.split('.')[0], interval_file_list))
                name_list.sort()

                # get interval specific data
                with open(osp.join(interval_path, 'info.pkl'), 'rb') as p:
                    interval_data = pickle.load(p, encoding='latin1')
                    K = interval_data['calib']['depth_intrinsic']
                    Rt_list = interval_data['pose']
                    camera_center_list = list(map(lambda Rt: -np.dot(Rt[:3, :3].T, Rt[:3, 3]), Rt_list))
                    print(seq_path + "/" + str(interval))
                    f = K[0, 0]
                    p_x = K[0, 2]
                    p_y = K[1, 2]

                img_list = []
                for f_name in name_list:
                    img_list.append(IO.read(osp.join(root_path, seq_path, interval, "raw_color", f_name + ".png")))

                plot_camera_sequence(camera_center_list)
                plot_frustum(Rt_list, img_list, f)

                if viewer.is_quit("n - Show next, q - Quit: "):
                    sys.exit(0)

    print('Dataset creation finished')
