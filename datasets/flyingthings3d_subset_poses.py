import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.utils.data as data
from utils import rigidity

__all__ = ['FlyingThings3DSubsetPoses']


class FlyingThings3DSubsetPoses(data.Dataset):
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
        self.root = osp.join(args.data_root, 'FlyingThings3D_subset_processed_35m_poses')
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
        pc1, pc2, R_rel, t_rel, sf_nr, sf_total = self.transform([pc1, pc2, R_rel, t_rel])
        if pc1 is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1, pc2, sf_nr, generated_data = self.gen_func([pc1, pc2, sf_nr], layer_num=3)
        sf_total = sf_total.T

        rot_rel = rigidity.torch_mat2euler(torch.tensor(R_rel))

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
        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(useful_paths) == 19586)
            else:
                assert (len(useful_paths) == 3816)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))

        Rt_rel = np.load(osp.join(path, 'Rt_rel.npy'))
        R_rel = Rt_rel[:3, :3]
        t_rel = Rt_rel[:3, 3]

        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        # Transform R and t accordingly
        negate_XZ = np.diag([-1, 1, -1]).astype(np.float32)
        R_rel = np.dot(np.dot(negate_XZ, R_rel), negate_XZ)
        t_rel = np.dot(negate_XZ, t_rel)
        return pc1, pc2, R_rel, t_rel
