import pickle
import sys
import os
import os.path as osp
from multiprocessing import Pool
import argparse

import IO
import IO_refresh
import numpy as np
from flyingthings3d_utils import *
import trimesh
import scipy.ndimage
from skimage.morphology import square, binary_erosion
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, help="path to the raw data")
parser.add_argument('--save_path', type=str, help="save path")
parser.add_argument('--only_save_near_pts', dest='save_near', action='store_true',
                    help='only save near points to save disk space')
parser.add_argument('--subset', type=int, help='divides the full dataset by N', default=-1)

args = parser.parse_args()
root_path = args.raw_data_path
save_path = args.save_path


def forward_backward_consistency(F_f, F_b, threshold):
    """
        IMPORTANT: the function actually performs B -> F -> B consistency check

        get the mask that is foreward-backward consistent
        Original code from Zhaoyang Lv, Learning Rigidity
    """
    u_b = F_b[0]
    v_b = F_b[1]
    u_f = F_f[0]
    v_f = F_f[1]
    [H, W] = np.shape(u_b)
    [x, y] = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x2 = x + u_b
    y2 = y + v_b
    # Out of boundary
    B = (x2 > W-1) | (y2 > H-1) | (x2 < 0) | (y2 < 0)
    u = scipy.ndimage.map_coordinates(u_f, [y2, x2])
    v = scipy.ndimage.map_coordinates(v_f, [y2, x2])
    u_inv = u
    v_inv = v

    dif = ((u_b + u_inv)**2 + (v_b + v_inv)**2)**0.5
    mask = (dif < threshold)
    mask = mask | B

    return mask


def export_ply(points, filepath):
    pc = trimesh.Trimesh(vertices=points)
    pc.export(filepath, vertex_normal=False)


def next_pixel2pc(flow, depth, f, cx, cy):
    height, width = depth.shape
    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx + flow[0]) * depth / f)[:, :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy + flow[1]) * depth / f)[:, :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)
    return pc


def pixel2pc(depth, f, cx, cy):
    height, width = depth.shape
    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx) * depth / f)[:, :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy) * depth / f)[:, :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)
    return pc


def process_frame_pair(params):
    """
    Process one frame pair and output PCs in np format.

    :param params: (IN/OUT Path, frame_id1, frame_id2, K)
    ('office3/keyframe_5',
    ('0700_0752', '000050'),
    ('0700_0752', '000051'),
    array([[583.,   0., 320.,   0.],
        [  0., 583., 240.,   0.],
        [  0.,   0.,   1.,   0.],
        [  0.,   0.,   0.,   1.]]))
    :return:
    """
    try:
        scene_root, frame_id1, frame_id2, camera_intrinsics, Rt_tuple = params
        interval1, frame_name1 = frame_id1
        interval2, frame_name2 = frame_id2
        Rt1 = Rt_tuple[0]
        Rt2 = Rt_tuple[1]

        out_filename = int(interval1.split('_')[0]) + int(frame_name1)
        out_filename = str(out_filename).zfill(6)
        save_folder_path = osp.join(save_path, scene_root, out_filename)
        os.makedirs(save_folder_path, exist_ok=True)

        f = camera_intrinsics[0, 0]
        p_x = camera_intrinsics[0, 2]
        p_y = camera_intrinsics[1, 2]

        # load depth
        depth1 = IO_refresh.pngdepth_read(osp.join(root_path, scene_root, interval1, "depth", frame_name1 + ".png"))
        depth2 = IO_refresh.pngdepth_read(osp.join(root_path, scene_root, interval2, "depth", frame_name2 + ".png"))
        # load forward flow
        flow1_forward = IO_refresh.flow_read_from_flo(osp.join(root_path, scene_root, interval1, "flow_forward", frame_name1 + ".flo"))
        flow2_forward = IO_refresh.flow_read_from_flo(osp.join(root_path, scene_root, interval2, "flow_forward", frame_name2 + ".flo"))
        # load backward flow
        flow1_backward = IO_refresh.flow_read_from_flo(osp.join(root_path, scene_root, interval1, "flow_backward", frame_name1 + ".flo"))
        flow2_backward = IO_refresh.flow_read_from_flo(osp.join(root_path, scene_root, interval2, "flow_backward", frame_name2 + ".flo"))
        # load invalidity masks
        invalid_render1 = IO.read(osp.join(root_path, scene_root, interval1, "invalid", frame_name1 + ".png"))
        invalid_render2 = IO.read(osp.join(root_path, scene_root, interval2, "invalid", frame_name2 + ".png"))

        # reconstruct Z -> X,Y,Z. RHS x right ,y down, z inside
        pc1 = pixel2pc(depth1, f, p_x, p_y)
        pc2 = next_pixel2pc(flow1_forward, depth2, f, p_x, p_y)

        # estimate occlusion mask as in Learning Rigidity
        # B -> F -> B
        occlusion_mask1 = forward_backward_consistency(flow1_forward, flow2_backward, 0.1)
        occlusion_mask1 = binary_erosion(occlusion_mask1, square(10))
        # F -> B -> F
        occlusion_mask2 = forward_backward_consistency(flow2_backward, flow1_forward, 0.1)
        occlusion_mask2 = binary_erosion(occlusion_mask2, square(10))
        # OCC mask estimate
        occlusion_mask = np.logical_and(occlusion_mask1, occlusion_mask2)

        # create validity mask
        valid_render = np.logical_and(invalid_render1 == 0, invalid_render2 == 0)
        valid_mask = np.logical_and(valid_render, occlusion_mask)
        # IO.write(osp.join(save_folder_path, 'valid_mask.png'), valid_mask.astype(np.uint8) * 255)
        pc1 = pc1[valid_mask]
        pc2 = pc2[valid_mask]

        # remove all points with huge scene flows
        sf_norm = np.linalg.norm(pc2 - pc1, axis=1)

        # Distribution of magnitudes
        # plt.hist(sf_norm, bins=10000)
        # plt.show()

        small_depth_diff = sf_norm < np.mean(sf_norm) + 2*np.std(sf_norm)
        pc1 = pc1[small_depth_diff]
        pc2 = pc2[small_depth_diff]

        # Distribution of magnitudes after filtering
        # sf_norm = np.linalg.norm(pc2 - pc1, axis=1)
        # plt.hist(sf_norm, bins=10000)
        # plt.show()

        # Ego warping
        R1 = Rt1[:3, :3]
        R2 = Rt2[:3, :3]
        t1 = Rt1[:3, 3]
        t2 = Rt2[:3, 3]
        # compute camera motion
        R_rel = np.dot(R2.T, R1)
        t_rel = np.dot(R2.T, t1 - t2)
        # Persist Relative transform from C1 -> C2
        Rt_rel = np.c_[R_rel, t_rel]
        Rt_rel = np.r_[Rt_rel, np.array([[0, 0, 0, 1]])]
        np.save(osp.join(save_folder_path, 'Rt_rel.npy'), Rt_rel.astype(np.float32))

        if not args.save_near:
            np.save(osp.join(save_folder_path, 'pc1.npy'), pc1)
            np.save(osp.join(save_folder_path, 'pc2.npy'), pc2)
        else:
            near_mask = np.logical_and(pc1[..., -1] < 35., pc2[..., -1] < 35.)
            np.save(osp.join(save_folder_path, 'pc1.npy'), pc1[near_mask])
            np.save(osp.join(save_folder_path, 'pc2.npy'), pc2[near_mask])

    except Exception as ex:
        print('error in addressing params', params, 'see exception:')
        print(ex)
        sys.stdout.flush()
        return


scene_list = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
keyframe_list = ['keyframe_1']

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
            # Sorting not strictly needed unless you do index intervals_list[i] & intervals_list[i+1]
            intervals_list.sort()
            # loop over intervals
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

                # loop over N - 1 frames inside an interval
                for j in range(0, len(name_list) - 1):
                    f_name1 = name_list[j]
                    f_name2 = name_list[j + 1]
                    Rt1 = Rt_list[j]
                    Rt2 = Rt_list[j + 1]
                    param_list.append((seq_path, (interval, f_name1), (interval, f_name2), K, (Rt1, Rt2)))

    if args.subset != -1:
        param_list = param_list[::args.subset]

    pool = Pool()
    pool.map(process_frame_pair, param_list)
    pool.close()
    pool.join()

    print('Dataset creation finished')
