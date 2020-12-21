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


split_lookup = {"TRAIN": "train", "TEST": "val"}

for split in splits:
    split_count = 0

    out_left = osp.join(output_root, split_lookup[split], 'camera_poses', 'left')
    out_right = osp.join(output_root, split_lookup[split], 'camera_poses', 'right')
    os.makedirs(out_left, exist_ok=True)
    os.makedirs(out_right, exist_ok=True)

    with open(split + '.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            camera_path = osp.join(pose_root, line.split(' ')[0], 'camera_data.txt')
            frame_number = line.split(' ')[1]

            frame_matched = False

            with open(camera_path, 'r') as camera_f:
                for camera_line in camera_f:
                    if camera_line.rstrip().startswith("Frame " + frame_number):
                        left_T = parse_pose(next(camera_f).rstrip())
                        right_T = parse_pose(next(camera_f).rstrip())

                        out_name = str(split_count).zfill(7)
                        np.save(osp.join(out_left, out_name + '.npy'), left_T)
                        np.save(osp.join(out_right, out_name + '.npy'), right_T)
                        split_count += 1

                        frame_matched = True
                        break

            if not frame_matched:
                split_count += 1
                print(camera_path)
                print(line)
                print("")
