import os
import os.path as osp

target_dir = '/home/ivan/Downloads/frames_finalpass_webp/'

splits = ["TEST", "TRAIN"]
subsets = ["A", "B", "C"]


for split in splits:
    split_count = 0
    with open(split + '.txt', 'w') as f:
        for subset in subsets:
            subset_root = osp.join(target_dir, split, subset)
            scene_list = os.listdir(subset_root)
            scene_list.sort()
            for scene in scene_list:
                scene_root = osp.join(subset_root, scene, "left")
                img_list = os.listdir(scene_root)
                img_list.sort()
                for img in img_list:
                    if os.path.isfile(osp.join(scene_root, img)):
                        f.write(osp.join(split, subset, scene) + ' ' + str(int(img.split('.')[0])) + '\n')
                        split_count += 1
    print(split_count)
