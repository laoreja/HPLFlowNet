import os, sys
import os.path as osp
from collections import defaultdict
import numbers
import math
import numpy as np
import traceback
import time

import torch

import numba
from numba import njit, cffi_support

from . import functional as F

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'models'))
import _khash_ffi

cffi_support.register_module(_khash_ffi)
khash_init = _khash_ffi.lib.khash_int2int_init
khash_get = _khash_ffi.lib.khash_int2int_get
khash_set = _khash_ffi.lib.khash_int2int_set
khash_destroy = _khash_ffi.lib.khash_int2int_destroy


# ---------- BASIC operations ----------
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            return pic
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ---------- Build permutalhedral lattice ----------
@njit(numba.int64(numba.int64[:], numba.int64, numba.int64[:], numba.int64[:], ))
def key2int(key, dim, key_maxs, key_mins):
    """
    :param key: np.array
    :param dim: int
    :param key_maxs: np.array
    :param key_mins: np.array
    :return:
    """
    tmp_key = key - key_mins
    scales = key_maxs - key_mins + 1
    res = 0
    for idx in range(dim):
        res += tmp_key[idx]
        res *= scales[idx + 1]
    res += tmp_key[dim]
    return res


@njit(numba.int64[:](numba.int64, numba.int64, numba.int64[:], numba.int64[:], ))
def int2key(int_key, dim, key_maxs, key_mins):
    key = np.empty((dim + 1,), dtype=np.int64)
    scales = key_maxs - key_mins + 1
    for idx in range(dim, 0, -1):
        key[idx] = int_key % scales[idx]
        int_key -= key[idx]
        int_key //= scales[idx]
    key[0] = int_key

    key += key_mins
    return key


@njit
def advance_in_dimension(d1, increment, adv_dim, key):
    key_cp = key.copy()

    key_cp -= increment
    key_cp[adv_dim] += increment * d1
    return key_cp


class Traverse:
    def __init__(self, neighborhood_size, d):
        self.neighborhood_size = neighborhood_size
        self.d = d

    def go(self, start_key, hash_table_list):
        walking_keys = np.empty((self.d + 1, self.d + 1), dtype=np.long)
        self.walk_cuboid(start_key, 0, False, walking_keys, hash_table_list)

    def walk_cuboid(self, start_key, d, has_zero, walking_keys, hash_table_list):
        if d <= self.d:
            walking_keys[d] = start_key.copy()

            range_end = self.neighborhood_size + 1 if (has_zero or (d < self.d)) else 1
            for i in range(range_end):
                self.walk_cuboid(walking_keys[d], d + 1, has_zero or (i == 0), walking_keys, hash_table_list)
                walking_keys[d] = advance_in_dimension(self.d + 1, 1, d, walking_keys[d])
        else:
            hash_table_list.append(start_key.copy())


@njit
def build_unsymmetric(pc1_num_points, pc2_num_points,
                      d1, bcn_filter_size, corr_filter_size, corr_corr_size,
                      pc1_keys_np, pc2_keys_np,
                      key_maxs, key_mins,
                      pc1_lattice_offset, pc2_lattice_offset,
                      bcn_filter_offsets,
                      pc1_blur_neighbors, pc2_blur_neighbors,
                      corr_filter_offsets, corr_corr_offsets,
                      pc1_corr_indices, pc2_corr_indices,
                      last_pc1, last_pc2,
                      assign_last):
    """

    :param pc1_num_points: int. Given
    :param pc2_num_points: int. Given
    :param d1: int. Given
    :param bcn_filter_size: int. Given. -1 indicates "do not filter"
    :param corr_filter_size: int. Displacement filtering radius. Given. -1 indicates "do not filter"
    :param corr_corr_size: int. Patch correlation radius. Given. -1 indicates "do not filter"
    :param pc1_keys_np: (d1, N, d1) long. Given. lattice points coordinates
    :param pc2_keys_np:
    :param key_maxs: (d1,) long. Given
    :param key_mins:
    :param pc1_lattice_offset: (d1, N) long. hash indices for pc1_keys_np
    :param pc2_lattice_offset:
    :param bcn_filter_offsets: (bcn_filter_size, d1) long. Given.
    :param pc1_blur_neighbors: (bcn_filter_size, pc1_hash_cnt) long. hash indices. -1 means not in the hash table
    :param pc2_blur_neighbors: (bcn_filter_size, pc2_hash_cnt)
    :param corr_filter_offsets: (corr_filter_size, d1) long. Given.
    :param corr_corr_offsets: (corr_corr_size, d1) long. Given.
    :param pc1_corr_indices: (corr_corr_size, pc1_hash_cnt) long. hash indices
    :param pc2_corr_indices: (corr_filter_size, corr_corr_size, pc1_hash_cnt) long. hash indices
    :param last_pc1: (d1, pc1_hash_cnt). permutohedral coordiantes for the next scale.
    :param last_pc2: (d1, pc2_hash_cnt)
    :return:
    """
    # build hash table
    hash_table1 = khash_init()  # key to hash index
    key_hash_table1 = khash_init()  # hash index to key
    hash_table2 = khash_init()
    if bcn_filter_size != -1:
        key_hash_table2 = khash_init()

    hash_cnt1 = 0
    hash_cnt2 = 0
    for point_idx in range(pc1_num_points):
        for remainder in range(d1):
            key_int1 = key2int(pc1_keys_np[:, point_idx, remainder], d1 - 1, key_maxs, key_mins)
            hash_idx1 = khash_get(hash_table1, key_int1, -1)
            if hash_idx1 == -1:
                # insert lattice into hash table
                khash_set(hash_table1, key_int1, hash_cnt1)
                khash_set(key_hash_table1, hash_cnt1, key_int1)
                hash_idx1 = hash_cnt1
                if assign_last:
                    last_pc1[:, hash_idx1] = pc1_keys_np[:, point_idx, remainder]

                hash_cnt1 += 1
            pc1_lattice_offset[remainder, point_idx] = hash_idx1

    for point_idx in range(pc2_num_points):
        for remainder in range(d1):
            key_int2 = key2int(pc2_keys_np[:, point_idx, remainder], d1 - 1, key_maxs, key_mins)
            hash_idx2 = khash_get(hash_table2, key_int2, -1)
            if hash_idx2 == -1:
                khash_set(hash_table2, key_int2, hash_cnt2)
                if bcn_filter_size != -1:
                    khash_set(key_hash_table2, hash_cnt2, key_int2)
                hash_idx2 = hash_cnt2
                if assign_last:
                    last_pc2[:, hash_idx2] = pc2_keys_np[:, point_idx, remainder]

                hash_cnt2 += 1
            pc2_lattice_offset[remainder, point_idx] = hash_idx2

    for hash_idx in range(hash_cnt1):
        pc1_int_key = khash_get(key_hash_table1, hash_idx, -1)
        pc1_key = int2key(pc1_int_key, d1 - 1, key_maxs, key_mins)

        if bcn_filter_size != -1:
            neighbor_keys = pc1_key + bcn_filter_offsets  # (#pts in the filter, d)
            for bcn_filter_index in range(bcn_filter_size):
                pc1_blur_neighbors[bcn_filter_index, hash_idx] = khash_get(hash_table1,
                                                                           key2int(neighbor_keys[bcn_filter_index, :],
                                                                                   d1 - 1,
                                                                                   key_maxs,
                                                                                   key_mins),
                                                                           -1)

        if corr_filter_size != -1:
            corr_pc1_keys = pc1_key + corr_corr_offsets  # (#pts in the filter, d)
            for corr_index in range(corr_corr_size):
                corr_pc1_key = corr_pc1_keys[corr_index, :]
                pc1_corr_indices[corr_index, hash_idx] = khash_get(hash_table1,
                                                                   key2int(corr_pc1_key, d1 - 1,
                                                                           key_maxs,
                                                                           key_mins),
                                                                   -1)

                corr_pc2_keys = corr_pc1_key + corr_filter_offsets
                for filter_index in range(corr_filter_size):
                    corr_pc2_key = corr_pc2_keys[filter_index, :]
                    pc2_corr_indices[filter_index, corr_index, hash_idx] = khash_get(hash_table2,
                                                                                     key2int(corr_pc2_key,
                                                                                             d1 - 1,
                                                                                             key_maxs,
                                                                                             key_mins),
                                                                                     -1)

    if bcn_filter_size != -1:
        for hash_idx in range(hash_cnt2):
            pc2_int_key = khash_get(key_hash_table2, hash_idx, -1)
            pc2_key = int2key(pc2_int_key, d1 - 1, key_maxs, key_mins)

            neighbor_keys = pc2_key + bcn_filter_offsets
            for bcn_filter_index in range(bcn_filter_size):
                pc2_blur_neighbors[bcn_filter_index, hash_idx] = khash_get(hash_table2,
                                                                           key2int(neighbor_keys[bcn_filter_index, :],
                                                                                   d1 - 1,
                                                                                   key_maxs,
                                                                                   key_mins),
                                                                           -1)
    # destroy hash table
    khash_destroy(hash_table1)
    khash_destroy(key_hash_table1)
    khash_destroy(hash_table2)
    if bcn_filter_size != -1:
        khash_destroy(key_hash_table2)


class GenerateDataUnsymmetric(object):
    def __init__(self, args):
        self.d = args.dim
        d = args.dim
        self.d1 = self.d + 1
        self.scales_filter_map = args.scales_filter_map

        elevate_left = torch.ones((self.d1, self.d), dtype=torch.float32).triu()
        elevate_left[1:, ] += torch.diag(torch.arange(-1, -d - 1, -1, dtype=torch.float32))
        elevate_right = torch.diag(1. / (torch.arange(1, d + 1, dtype=torch.float32) *
                                         torch.arange(2, d + 2, dtype=torch.float32)).sqrt())
        self.expected_std = (d + 1) * math.sqrt(2 / 3)
        self.elevate_mat = torch.mm(elevate_left, elevate_right)
        # (d+1,d)
        del elevate_left, elevate_right

        # canonical
        canonical = torch.arange(d + 1, dtype=torch.long)[None, :].repeat(d + 1, 1)
        # (d+1, d+1)
        for i in range(1, d + 1):
            canonical[-i:, i] = i - d - 1
        self.canonical = canonical

        self.dim_indices = torch.arange(d + 1, dtype=torch.long)[:, None]

        self.radius2offset = {}
        radius_set = set([item for line in self.scales_filter_map for item in line[1:] if item != -1])

        for radius in radius_set:
            hash_table = []
            center = np.array([0] * self.d1, dtype=np.long)

            traversal = Traverse(radius, self.d)
            traversal.go(center, hash_table)
            self.radius2offset[radius] = np.vstack(hash_table)

    def get_keys_and_barycentric(self, pc):
        """

        :param pc: (self.d, N -- undefined)
        :return:
        """
        num_points = pc.size(-1)
        point_indices = torch.arange(num_points, dtype=torch.long)[None, :]

        elevated = torch.matmul(self.elevate_mat, pc) * self.expected_std  # (d+1, N)

        # find 0-remainder
        greedy = torch.round(elevated / self.d1) * self.d1  # (d+1, N)

        el_minus_gr = elevated - greedy
        rank = torch.sort(el_minus_gr, dim=0, descending=True)[1]
        # the following advanced indexing is different in PyTorch 0.4.0 and 1.0.0
        #rank[rank, point_indices] = self.dim_indices  # works in PyTorch 0.4.0 but fail in PyTorch 1.x
        index = rank.clone()
        rank[index, point_indices] = self.dim_indices  # works both in PyTorch 1.x(has tested in PyTorch 1.2) and PyTorch 0.4.0
        del index

        remainder_sum = greedy.sum(dim=0, keepdim=True) / self.d1

        rank_float = rank.type(torch.float32)
        cond_mask = ((rank_float >= self.d1 - remainder_sum) * (remainder_sum > 0) + \
                     (rank_float < -remainder_sum) * (remainder_sum < 0)) \
            .type(torch.float32)
        sum_gt_zero_mask = (remainder_sum > 0).type(torch.float32)
        sum_lt_zero_mask = (remainder_sum < 0).type(torch.float32)
        sign_mask = -1 * sum_gt_zero_mask + sum_lt_zero_mask

        greedy += self.d1 * sign_mask * cond_mask
        rank += (self.d1 * sign_mask * cond_mask).type_as(rank)
        rank += remainder_sum.type(torch.long)

        # barycentric
        el_minus_gr = elevated - greedy
        greedy = greedy.type(torch.long)

        barycentric = torch.zeros((self.d1 + 1, num_points), dtype=torch.float32)
        barycentric[self.d - rank, point_indices] += el_minus_gr
        barycentric[self.d1 - rank, point_indices] -= el_minus_gr
        barycentric /= self.d1
        barycentric[0, point_indices] += 1. + barycentric[self.d1, point_indices]
        barycentric = barycentric[:-1, :]

        keys = greedy[:, :, None] + self.canonical[rank, :]  # (d1, num_points, d1)
        # rank: rearrange the coordinates of the canonical

        keys_np = keys.numpy()
        del elevated, greedy, rank, remainder_sum, rank_float, \
            cond_mask, sum_gt_zero_mask, sum_lt_zero_mask, sign_mask
        return keys_np, barycentric, el_minus_gr

    def get_filter_size(self, radius):
        return (radius + 1) ** self.d1 - radius ** self.d1

    def __call__(self, data):
        pc1, pc2, sf = data
        if pc1 is None:
            return None, None, None, None

        with torch.no_grad():
            pc1 = torch.from_numpy(pc1.T)
            pc2 = torch.from_numpy(pc2.T)
            sf = torch.from_numpy(sf.T)

            generated_data = []
            last_pc1 = pc1.clone()
            last_pc2 = pc2.clone()
            pc1_num_points = pc1.size(-1)
            pc2_num_points = pc2.size(-1)

            for idx, (scale, bcn_filter_raidus, corr_filter_radius, corr_corr_radius) in enumerate(
                    self.scales_filter_map):

                last_pc1[:3, :] *= scale
                last_pc2[:3, :] *= scale

                pc1_keys_np, pc1_barycentric, pc1_el_minus_gr = self.get_keys_and_barycentric(last_pc1)
                pc2_keys_np, pc2_barycentric, pc2_el_minus_gr = self.get_keys_and_barycentric(last_pc2)
                # keys: (d1, N, d1) [[:, point_idx, remainder_idx]], barycentric: (d1, N), el_minus_gr: (d1, N)

                key_maxs = np.maximum(pc1_keys_np.max(-1).max(-1), pc2_keys_np.max(-1).max(-1))
                key_mins = np.minimum(pc1_keys_np.min(-1).min(-1), pc2_keys_np.min(-1).min(-1))

                pc1_keys_set = set(map(tuple, pc1_keys_np.reshape(self.d1, -1).T))
                pc2_keys_set = set(map(tuple, pc2_keys_np.reshape(self.d1, -1).T))

                pc1_hash_cnt = len(pc1_keys_set)
                pc2_hash_cnt = len(pc2_keys_set)

                pc1_lattice_offset = np.empty((self.d1, pc1_num_points), dtype=np.int64)
                pc2_lattice_offset = np.empty((self.d1, pc2_num_points), dtype=np.int64)

                if bcn_filter_raidus != -1:
                    bcn_filter_size = self.get_filter_size(bcn_filter_raidus)
                    pc1_blur_neighbors = np.empty((bcn_filter_size, pc1_hash_cnt), dtype=np.int64)
                    pc1_blur_neighbors.fill(-1)
                    pc2_blur_neighbors = np.empty((bcn_filter_size, pc2_hash_cnt), dtype=np.int64)
                    pc2_blur_neighbors.fill(-1)
                    bcn_filter_offsets = self.radius2offset[bcn_filter_raidus]
                else:
                    bcn_filter_size = -1
                    pc1_blur_neighbors = np.zeros((1, 1), dtype=np.int64)
                    pc2_blur_neighbors = np.zeros((1, 1), dtype=np.int64)
                    bcn_filter_offsets = np.zeros((1, 1), dtype=np.int64)

                if corr_filter_radius != -1:
                    corr_filter_size = self.get_filter_size(corr_filter_radius)
                    corr_corr_size = self.get_filter_size(corr_corr_radius)
                    pc1_corr_indices = np.empty((corr_corr_size, pc1_hash_cnt), dtype=np.int64)
                    pc1_corr_indices.fill(-1)
                    pc2_corr_indices = np.empty((corr_filter_size, corr_corr_size, pc1_hash_cnt), dtype=np.int64)
                    pc2_corr_indices.fill(-1)
                    corr_filter_offsets, corr_corr_offsets = self.radius2offset[corr_filter_radius], \
                                                             self.radius2offset[corr_corr_radius]
                else:
                    corr_filter_size = -1
                    corr_corr_size = -1
                    pc1_corr_indices = np.zeros((1, 1, 1), dtype=np.int64)
                    pc2_corr_indices = np.zeros((1, 1, 1), dtype=np.int64)
                    corr_filter_offsets, corr_corr_offsets = np.zeros((1, 1), dtype=np.int64), np.zeros((1, 1),
                                                                                                        dtype=np.int64)
                if idx != len(self.scales_filter_map) - 1:
                    last_pc1 = np.empty((self.d1, pc1_hash_cnt), dtype=np.float32)
                    last_pc2 = np.empty((self.d1, pc2_hash_cnt), dtype=np.float32)
                else:
                    last_pc1 = np.zeros((1, 1), dtype=np.float32)
                    last_pc2 = np.zeros((1, 1), dtype=np.float32)

                build_unsymmetric(pc1_num_points, pc2_num_points,
                                  self.d1, bcn_filter_size, corr_filter_size, corr_corr_size,
                                  pc1_keys_np, pc2_keys_np,
                                  key_maxs, key_mins,
                                  pc1_lattice_offset, pc2_lattice_offset,
                                  bcn_filter_offsets,
                                  pc1_blur_neighbors, pc2_blur_neighbors,
                                  corr_filter_offsets, corr_corr_offsets,
                                  pc1_corr_indices, pc2_corr_indices,
                                  last_pc1, last_pc2,
                                  idx != len(self.scales_filter_map) - 1)

                pc1_lattice_offset = torch.from_numpy(pc1_lattice_offset)
                pc2_lattice_offset = torch.from_numpy(pc2_lattice_offset)

                if bcn_filter_size != -1:
                    pc1_blur_neighbors = torch.from_numpy(pc1_blur_neighbors)
                    pc2_blur_neighbors = torch.from_numpy(pc2_blur_neighbors)
                else:
                    pc1_blur_neighbors = torch.zeros(1, dtype=torch.long)
                    pc2_blur_neighbors = torch.zeros(1, dtype=torch.long)

                if corr_filter_size != -1:
                    pc1_corr_indices = torch.from_numpy(pc1_corr_indices)
                    pc2_corr_indices = torch.from_numpy(pc2_corr_indices)
                else:
                    pc1_corr_indices = torch.zeros(1, dtype=torch.long)
                    pc2_corr_indices = torch.zeros(1, dtype=torch.long)

                if idx != len(self.scales_filter_map) - 1:
                    last_pc1 = torch.from_numpy(last_pc1)
                    last_pc2 = torch.from_numpy(last_pc2)
                    last_pc1 /= self.expected_std * scale
                    last_pc2 /= self.expected_std * scale
                    last_pc1 = torch.matmul(self.elevate_mat.t(), last_pc1)
                    last_pc2 = torch.matmul(self.elevate_mat.t(), last_pc2)
                    pc1_num_points = pc1_hash_cnt
                    pc2_num_points = pc2_hash_cnt

                generated_data.append({'pc1_barycentric': pc1_barycentric,
                                       'pc2_barycentric': pc2_barycentric,
                                       'pc1_el_minus_gr': pc1_el_minus_gr,
                                       'pc2_el_minus_gr': pc2_el_minus_gr,
                                       'pc1_lattice_offset': pc1_lattice_offset,
                                       'pc2_lattice_offset': pc2_lattice_offset,
                                       'pc1_blur_neighbors': pc1_blur_neighbors,
                                       'pc2_blur_neighbors': pc2_blur_neighbors,
                                       'pc1_corr_indices': pc1_corr_indices,
                                       'pc2_corr_indices': pc2_corr_indices,
                                       'pc1_hash_cnt': pc1_hash_cnt,
                                       'pc2_hash_cnt': pc2_hash_cnt,
                                       })

            return pc1, pc2, sf, generated_data

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(scales_filter_map: {}\n'.format(self.scales_filter_map)
        format_string += ')'
        return format_string


# ---------- MAIN operations ----------
class ProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2 = data
        if pc1 is None:
            return None, None, None,

        sf = pc2[:, :3] - pc1[:, :3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string


class Augmentation(object):
    def __init__(self, aug_together_args, aug_pc2_args, data_process_args, num_points, allow_less_points=False):
        self.together_args = aug_together_args
        self.pc2_args = aug_pc2_args
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2 = data
        if pc1 is None:
            return None, None, None

        # together, order: scale, rotation, shift, jitter
        # scale
        scale = np.diag(np.random.uniform(self.together_args['scale_low'],
                                          self.together_args['scale_high'],
                                          3).astype(np.float32))
        # rotation
        angle = np.random.uniform(-self.together_args['degree_range'],
                                  self.together_args['degree_range'])
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rot_matrix = np.array([[cosval, 0, sinval],
                               [0, 1, 0],
                               [-sinval, 0, cosval]], dtype=np.float32)
        matrix = scale.dot(rot_matrix.T)

        # shift
        shifts = np.random.uniform(-self.together_args['shift_range'],
                                   self.together_args['shift_range'],
                                   (1, 3)).astype(np.float32)

        # jitter
        jitter = np.clip(self.together_args['jitter_sigma'] * np.random.randn(pc1.shape[0], 3),
                         -self.together_args['jitter_clip'],
                         self.together_args['jitter_clip']).astype(np.float32)
        bias = shifts + jitter

        pc1[:, :3] = pc1[:, :3].dot(matrix) + bias
        pc2[:, :3] = pc2[:, :3].dot(matrix) + bias

        # pc2, order: rotation, shift, jitter
        # rotation
        angle2 = np.random.uniform(-self.pc2_args['degree_range'],
                                   self.pc2_args['degree_range'])
        cosval2 = np.cos(angle2)
        sinval2 = np.sin(angle2)
        matrix2 = np.array([[cosval2, 0, sinval2],
                            [0, 1, 0],
                            [-sinval2, 0, cosval2]], dtype=pc1.dtype)
        # shift
        shifts2 = np.random.uniform(-self.pc2_args['shift_range'],
                                    self.pc2_args['shift_range'],
                                    (1, 3)).astype(np.float32)

        pc2[:, :3] = pc2[:, :3].dot(matrix2.T) + shifts2
        sf = pc2[:, :3] - pc1[:, :3]

        if not self.no_corr:
            jitter2 = np.clip(self.pc2_args['jitter_sigma'] * np.random.randn(pc1.shape[0], 3),
                              -self.pc2_args['jitter_clip'],
                              self.pc2_args['jitter_clip']).astype(np.float32)
            pc2[:, :3] += jitter2

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)

        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        pc2 = pc2[sampled_indices2]
        sf = sf[sampled_indices1]

        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(together_args: \n'
        for key in sorted(self.together_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.together_args[key])
        format_string += '\npc2_args: \n'
        for key in sorted(self.pc2_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.pc2_args[key])
        format_string += '\ndata_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
