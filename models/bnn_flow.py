import torch
import torch.nn as nn

from .bilateralNN import sparse_sum
from .module_utils import Conv2dReLU, Conv3dReLU

DELETE_TMP_VARIABLES = False


class BilateralCorrelationFlex(nn.Module):
    def __init__(self, d,
                 corr_filter_radius, corr_corr_radius,
                 num_input, num_corr_output, num_output,
                 DEVICE,
                 use_bias,
                 use_leaky,
                 use_norm,
                 prev_corr_dim,
                 last_relu,
                 chunk_size=1024 * 1024 * 25):
        """

        :param d: int (in our case, 3)
        :param corr_filter_radius: int
        :param corr_corr_radius: int
        :param num_input: int (C_in)
        :param num_corr_output: list of ints
        :param num_output: list of ints
        :param DEVICE: str, 'cuda' or whatever
        :param use_bias: bool. used after slicing, never used in this implementation.
        :param use_leaky: bool. used for conv modules
        :param use_norm: bool. whether to use our normalization scheme, always set it to be true for better performance.
        :param prev_corr_dim: int.
        """
        super(BilateralCorrelationFlex, self).__init__()

        self.d = d
        self.d1 = d + 1
        self.corr_size = self.get_filter_size(corr_corr_radius)
        self.filter_size = self.get_filter_size(corr_filter_radius)
        self.num_input = num_input
        self.prev_corr_dim = prev_corr_dim
        self.num_output = num_output
        self.DEVICE = DEVICE
        # self.use_bias = use_bias
        self.use_norm = use_norm
        self.last_relu = last_relu
        self.MAX_SIZE = chunk_size  # 1024 * 1024 * 25

        # define needed buffers
        self.register_buffer('feat_indices', torch.arange(num_input, dtype=torch.long))
        if prev_corr_dim != 0:
            self.register_buffer('feat1_indices', torch.arange(num_input + prev_corr_dim, dtype=torch.long))
        else:
            self.feat1_indices = self.feat_indices
        num_final_output = num_output[-1]
        self.register_buffer('out_indices', torch.arange(num_final_output, dtype=torch.long))

        # define corr conv modules (patch correlation)
        corr_sequential_list = []
        n_in_channel = num_input * 2 + prev_corr_dim
        for idx, n_out_channel in enumerate(num_corr_output):
            if idx == 0:
                kernel_size = (1, self.corr_size, 1)
            else:
                kernel_size = (1, 1, 1)
            corr_sequential_list.append(Conv3dReLU(n_in_channel, n_out_channel, kernel_size, use_leaky=use_leaky))
            n_in_channel = n_out_channel
        self.corr_conv = nn.Sequential(*corr_sequential_list)

        # define filter conv modules (displacement filtering)
        filter_sequential_list = []
        n_in_channel = num_corr_output[-1]
        for idx, n_out_channel in enumerate(num_output[:-1]):
            if idx == 0:
                kernel_size = (self.filter_size, 1)
            else:
                kernel_size = (1, 1)
            filter_sequential_list.append(Conv2dReLU(n_in_channel, n_out_channel, kernel_size, use_leaky=use_leaky))
            n_in_channel = n_out_channel

        if len(num_output) == 1:
            kernel_size = (self.filter_size, 1)
        else:
            kernel_size = (1, 1)
        if not self.last_relu:
            filter_sequential_list.append(nn.Conv2d(n_in_channel, num_final_output, kernel_size=kernel_size))
        else:
            filter_sequential_list.append(
                Conv2dReLU(n_in_channel, num_final_output, kernel_size=kernel_size, use_leaky=use_leaky))
        self.blur_conv = nn.Sequential(*filter_sequential_list)

    def get_filter_size(self, dist):
        return (dist + 1) ** self.d1 - dist ** self.d1

    def forward(self, feat1, feat2, prev_corr_feat,
                barycentric1, lattice_offset1,
                pc1_corr_indices, pc2_corr_indices,
                max_hash_cnt1, max_hash_cnt2):
        """

        :param feat1: float (B, C, max_hash_cnt1)
        :param feat2: float (B, C, max_hash_cnt2)
        :param prev_corr_feat: float (B, C', N_in)
                # need to splat prev_corr_feat to the new scale and vertices
        :param barycentric1: float (B, d1, N_in)
        :param lattice_offset1: int64 (B, d1, N_in)
        :param pc1_corr_indices: int64 (B, corr_corr_size, max_hash_cnt1)
        :param pc2_corr_indices: int64 (B, corr_filter_size, corr_corr_size, max_hash_cnt1)
        :param max_hash_cnt1: int
        :param max_hash_cnt2: int
        :return:
        """
        batch_size = feat1.size(0)
        batch_indices = torch.arange(batch_size, dtype=torch.long)
        if self.DEVICE == 'cuda':
            batch_indices = batch_indices.pin_memory().cuda(non_blocking=True)

        if prev_corr_feat is not None:
            # -------------------- SPLAT --------------------
            # barycentric: (B, 1, d1, N), features: (B, feat_size, 1, N)
            tmp1 = (barycentric1[:, None, :, :] * prev_corr_feat[:, :, None, :]).permute(1, 0, 2, 3) \
                .reshape(self.prev_corr_dim, -1)
            # (B, feat_size, d1, N) -> (feat_size, B * d1 * N)
            tmp1 = tmp1.t()

            # plus one makes the first element of splatted is 0 in all channels
            prev_splatted1 = sparse_sum((lattice_offset1 + 1).reshape(1, -1), tmp1,
                                        torch.Size([batch_size * (max_hash_cnt1 + 1), self.prev_corr_dim]),
                                        self.DEVICE == 'cuda')
            prev_splatted1 = prev_splatted1.reshape(batch_size, max_hash_cnt1 + 1, self.prev_corr_dim).permute(0, 2, 1)

            if self.use_norm:
                # for density normalization
                one_feat1 = torch.ones((batch_size, 1, prev_corr_feat.size(-1)), dtype=torch.float32)
                if self.DEVICE == 'cuda':
                    one_feat1 = one_feat1.pin_memory().cuda(non_blocking=True)

                # (B, d1, N), (B, 1=feat_size, N) -> (B, d1, N)
                one_tmp1 = (barycentric1 * one_feat1).reshape(1, -1)  # (1, B * d1 * N)
                one_tmp1 = one_tmp1.t()

                # FLOP (B=1): d1 * N_in_prev
                one_splatted1 = sparse_sum((lattice_offset1 + 1).reshape(1, -1), one_tmp1,
                                           torch.Size([batch_size * (max_hash_cnt1 + 1), 1]),
                                           self.DEVICE == 'cuda')
                one_splatted1 = one_splatted1.reshape(batch_size, max_hash_cnt1 + 1)

                # print('normalize!')
                norm1 = 1. / (one_splatted1 + 1e-5)
                prev_splatted1 *= norm1[:, None, :]

                if DELETE_TMP_VARIABLES:
                    del one_feat1, one_tmp1, one_splatted1

        splatted1 = torch.cat((torch.zeros((batch_size, self.num_input, 1),
                                           dtype=feat1.dtype,
                                           device=feat1.device),
                               feat1),
                              dim=-1)

        splatted2 = torch.cat((torch.zeros((batch_size, self.num_input, 1),
                                           dtype=feat2.dtype,
                                           device=feat2.device),
                               feat2),
                              dim=-1)
        if prev_corr_feat is not None:
            splatted1 = torch.cat((prev_splatted1, splatted1), dim=1)

        # -------------------- BLUR --------------------
        if self.MAX_SIZE == -1:
            chunk_size = max_hash_cnt1
        else:
            chunk_size = max(1,
                             min(self.MAX_SIZE // (
                                         self.num_input * 2 + self.prev_corr_dim) // self.filter_size // self.corr_size,
                                 max_hash_cnt1))

        num_chunks = (max_hash_cnt1 + chunk_size - 1) // chunk_size

        corr_blurred = []
        for cidx in range(num_chunks):
            start = cidx * chunk_size
            end = min(max_hash_cnt1, start + chunk_size)

            # splatted: (B, feat_size, max_hash_cnt+1)
            # pc1_corr_indices: (B, corr_corr_size, max_hash_cnt1)
            # spread_out1: (B, feat_size, corr_corr_size, chunk_size/max_hash_cnt1+1)
            spread_out1 = splatted1[batch_indices[:, None, None, None],
                                    self.feat1_indices[None, :, None, None],
                                    (pc1_corr_indices + 1)[:, None, :, start:end]]
            spread_out1 = spread_out1[:, :, None, :, :].repeat(1, 1, self.filter_size, 1, 1)

            # spread_out2: (B, feat_size, corr_filter_size, corr_corr_size, chunk_size/max_hash_cnt1+1)
            spread_out2 = splatted2[batch_indices[:, None, None, None, None],
                                    self.feat_indices[None, :, None, None, None],
                                    (pc2_corr_indices + 1)[:, None, :, :, start:end]]

            combined_input = torch.cat((spread_out1, spread_out2), dim=1)
            #  (B, 2*feat_size+prev_corr_size, corr_filter_size, corr_corr_size, chunk_size/max_hash_cnt1)

            correlated = self.corr_conv(combined_input).squeeze(
                3)  # (B, num_corr_output[-1], filter_size, 1--squeezed, chunk_size/max_hash_cnt1)

            corr_blurred_chunk = self.blur_conv(correlated).squeeze(2)
            # (B, num_output, 1--squeezed, chunk_size/max_hash_cnt1)
            corr_blurred.append(corr_blurred_chunk)
        corr_blurred = torch.cat(corr_blurred, dim=-1)  # (B, C_out, max_hash_cnt1)

        return corr_blurred


