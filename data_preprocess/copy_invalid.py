import pickle
import os
import os.path as osp
import argparse
import shutil

# Example usage:
# python3 copy_invalid.py \
#   --static /media/hdd2/github/RefRESH/data/RefRESH/BundleFusion/render/ \
#   --dynamic /media/hdd2/github/RefRESH/data/RefRESH/BundleFusion_dynamic/render/

parser = argparse.ArgumentParser()
parser.add_argument('--static', type=str, help="path to static data")
parser.add_argument('--dynamic', type=str, help="path to dynamic data")

args = parser.parse_args()
dynamic_path = args.dynamic
static_path = args.static

scene_list = ['apt0', 'apt1', 'apt2', 'copyroom', 'office0', 'office1', 'office2', 'office3']
keyframe_list = ['keyframe_1']

if __name__ == '__main__':
    for scene in scene_list:
        for keyframe in keyframe_list:
            seq_path = osp.join(scene, keyframe)
            interval_root = osp.join(dynamic_path, seq_path)
            intervals_list = os.listdir(interval_root)
            # loop over sorted intervals
            for interval in intervals_list:
                interval_path = osp.join(interval_root, interval)
                output_dir = osp.join(interval_path, 'invalid')
                os.makedirs(output_dir, exist_ok=True)
                # get interval specific data
                with open(osp.join(interval_path, 'info.pkl'), 'rb') as p:
                    interval_data = pickle.load(p, encoding='latin1')
                    invalid_list = interval_data['invalid']
                    depth_list = interval_data['depth']

                for j in range(len(invalid_list)):
                    invalid_path = '/'.join(invalid_list[j].split('/')[-4:])
                    in_path = osp.join(static_path, invalid_path)
                    out_path = osp.join(output_dir, depth_list[j].split('/')[-1])
                    shutil.copy(in_path, out_path)

    print('Copying invalid finished')
