import argparse
import os
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--poses', type=str, help="Path to the camera pose dir")
parser.add_argument('--output', type=str, help="Path to existing FlyingThings3D_subset_processed_35m")

args = parser.parse_args()
pose_root = args.poses
output_root = args.output

subsets = ["A", "B", "C"]
splits = ["TEST", "TRAIN"]

TARGET_NUM = 26066


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    fill_len = int(length * iteration // total)
    bar = fill * fill_len + '-' * (length - fill_len)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def parse_pose(pose_line):
    pose = []
    pose_str = pose_line.split(' ')
    for i in range(1, len(pose_str), 4):
        pose_row = [float(val) for val in pose_str[i:i + 4]]
        pose.append(pose_row)
    return np.array(pose)


omitted_scenes = set()
with open('all_unused_files.txt', 'r') as f:
    for line in f:
        omitted_scenes.add(line.rstrip())

total_counter = 0
for split in splits:
    frame_counter = 0
    out_split = 'train' if split == 'TRAIN' else 'val'
    out_left = osp.join(output_root, out_split, 'camera_poses', 'left')
    out_right = osp.join(output_root, out_split, 'camera_poses', 'right')
    os.makedirs(out_left, exist_ok=True)
    os.makedirs(out_right, exist_ok=True)
    for subset in subsets:
        subset_root = osp.join(pose_root, split, subset)
        scene_list = os.listdir(subset_root)
        scene_list.sort()
        for scene_idx, scene in enumerate(scene_list):
            scene_root = osp.join(subset_root, scene)
            with open(osp.join(scene_root, 'camera_data.txt'), 'r') as f:
                content = f.read().splitlines()

                for i in range(0, len(content) - 1, 4):
                    frame_num = content[i].split(' ')[-1].zfill(4)
                    check_name = osp.join(split, subset, scene, 'left', frame_num + '.png')
                    if check_name not in omitted_scenes:
                        left_pose = parse_pose(content[i + 1])
                        right_pose = parse_pose(content[i + 2])
                        out_filename = str(frame_counter).zfill(7)
                        np.save(osp.join(out_left, out_filename + '.npy'), left_pose)
                        np.save(osp.join(out_right, out_filename + '.npy'), right_pose)
                        print_progress(total_counter, TARGET_NUM, prefix='Progress:', suffix='Complete', length=50)
                        frame_counter += 1
                        total_counter += 1

assert total_counter == TARGET_NUM
