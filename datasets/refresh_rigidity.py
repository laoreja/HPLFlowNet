import sys
import os
import os.path as osp
import numpy as np
from transforms3d import euler

import torch.utils.data as data

__all__ = ['RefRESHRigidity']

VAL_SCENES = ["office3"]
TRAIN_SCENES = ["apt0", "apt1", "apt2", "copyroom", "office0", "office1", "office2"]
KEYFRAMES = ['keyframe_1']
TRAIN_SIZE = 40307
VAL_SIZE = 3720
DIVIDE_TRAIN = 8
DIVIDE_VAL = 4
INPUT_SCALAR = 10.


class RefRESHRigidity(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 gen_func,
                 args):
        self.root = osp.join(args.data_root, 'REFRESH_pc')
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points

        full = hasattr(args, 'full') and args.full
        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1, pc2, R_rel, t_rel = self.pc_loader(self.samples[index])
        # Augment and get SF
        pc1, pc2, R_rel, t_rel, sf_nr, sf_total = self.transform([pc1, pc2, R_rel, t_rel])

        if pc1 is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1, pc2, sf_nr, generated_data = self.gen_func([pc1, pc2, sf_nr])
        sf_total = sf_total.T

        # convert mat -> Euler angles
        rot_rel = np.array(euler.mat2euler(R_rel))

        return pc1, pc2, rot_rel, t_rel, sf_nr, sf_total, generated_data, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):
        root = osp.realpath(osp.expanduser(self.root))
        if self.train:
            scene_list = TRAIN_SCENES
            expected_size = TRAIN_SIZE
            select_scale = DIVIDE_TRAIN
        else:
            scene_list = VAL_SCENES
            expected_size = VAL_SIZE
            select_scale = DIVIDE_VAL

        useful_paths = []
        for scene in scene_list:
            for keyframe in KEYFRAMES:
                seq_root = osp.join(root, scene, keyframe)
                all_paths = os.walk(seq_root)
                useful_paths.extend(sorted([item[0] for item in all_paths if len(item[1]) == 0]))

        try:
            assert (len(useful_paths) == expected_size)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::select_scale]
        else:
            res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        # Loading of Rt_Rel
        Rt_rel = np.load(osp.join(path, 'Rt_rel.npy')).astype(np.float32)
        R_rel = Rt_rel[:3, :3]
        t_rel = Rt_rel[:3, 3]
        # Add X,Y axis inversion
        pc1[..., :2] *= -1.
        pc2[..., :2] *= -1.
        # Scale by INPUT_SCALAR
        pc1 *= INPUT_SCALAR
        pc2 *= INPUT_SCALAR
        # Transform R and t accordingly
        negate_XY = np.diag([-1, -1, 1]).astype(np.float32)
        R_rel = np.dot(np.dot(negate_XY, R_rel), negate_XY)
        t_rel = np.dot(negate_XY, t_rel) * INPUT_SCALAR
        return pc1, pc2, R_rel, t_rel
