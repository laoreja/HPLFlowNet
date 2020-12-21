import os
import os.path as osp

target_dir = '/home/ivan/Downloads/frames_finalpass_webp/'

hard_samples = list(open('all_unused_files.txt').read().splitlines())


for sample in hard_samples:
    webp_sample = sample[:-4] + '.webp'
    del_path = osp.join(target_dir, webp_sample)
    os.remove(del_path)
