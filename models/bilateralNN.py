import torch
import torch.nn as nn

from .module_utils import Conv2dReLU

DELETE_TMP_VARIABLES = False


class SparseSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, size, cuda):
        """

        :param ctx:
        :param indices: (1, B*d1*N)
        :param values: (B*d1*N, feat_size)
        :param size: (B*(H+1), feat_size)
        :param cuda: bool
        :return: (B*(H+1), feat_size)
        """

        ctx.save_for_backward(indices)

        if cuda:
            output = torch.cuda.sparse.FloatTensor(indices, values, size)
        else:
            output = torch.sparse.FloatTensor(indices, values, size)

        output = output.to_dense()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        grad_values = None
        if ctx.needs_input_grad[1]:
            grad_values = grad_output[indices.squeeze(0), :]

        return None, grad_values, None, None


sparse_sum = SparseSum.apply


class BilateralConvFlex(nn.Module):
    def __init__(self,
                 d, neighborhood_size,
                 num_input, num_output,
                 DEVICE,
                 use_bias,
                 use_leaky,
                 use_norm,
                 do_splat,
                 do_slice,
                 last_relu,
                 chunk_size=1024 * 1024 * 25):
        """

        :param d: int. Original dim of position (3 in our case)
        :param neighborhood_size: int.
        :param num_input: int. C_in for convolution.
        :param num_output: list. C_outs for convolution.
        :param DEVICE: str, "cuda" or whatever.
        :param use_bias: bool. Whether to use bias after slicing
        :param use_leaky: bool. Whether to use LeakyReLU
        :param use_norm: bool. Our normalization scheme. Always set it to be true for good performance.
        :param do_slice: bool.
        :param last_relu: bool. Whether to do relu for the last convolution layer in the blur/conv stage.
        :param chunk_size: int. max size for convolution, when set to be -1, no chunking operation.
        """
        super(BilateralConvFlex, self).__init__()

        self.d = d
        self.d1 = d + 1
        self.neighborhood_size = neighborhood_size
        self.filter_size = self.get_filter_size()
        self.num_input = num_input
        self.num_output = num_output
        self.DEVICE = DEVICE
        self.use_bias = use_bias  # only useful when do_slice = True
        self.do_splat = do_splat
        self.do_slice = do_slice
        self.last_relu = last_relu
        self.use_norm = use_norm
        self.MAX_SIZE = chunk_size  # 1024 * 1024 * 25

        num_final_output = num_output[-1]

        self.register_buffer('feat_indices', torch.arange(num_input, dtype=torch.long))
        if self.do_slice:
            self.register_buffer('out_indices', torch.arange(num_final_output, dtype=torch.long))

        sequential_list = []
        n_in_channel = num_input
        for idx, n_out_channel in enumerate(num_output[:-1]):
            if idx == 0:
                kernel_size = (self.filter_size, 1)
            else:
                kernel_size = (1, 1)
            sequential_list.append(Conv2dReLU(n_in_channel, n_out_channel, kernel_size, use_leaky=use_leaky))
            n_in_channel = n_out_channel

        if len(num_output) == 1:
            kernel_size = (self.filter_size, 1)
        else:
            kernel_size = (1, 1)
        if not self.last_relu:
            sequential_list.append(nn.Conv2d(n_in_channel, num_final_output, kernel_size=kernel_size))
        else:
            sequential_list.append(
                Conv2dReLU(n_in_channel, num_final_output, kernel_size=kernel_size, use_leaky=use_leaky))
        self.blur_conv = nn.Sequential(*sequential_list)

        if self.do_slice and self.use_bias:
            self.register_parameter('bias', nn.Parameter(data=torch.zeros((num_final_output,), dtype=torch.float32),
                                                         requires_grad=True))

    def get_filter_size(self):
        return (self.neighborhood_size + 1) ** self.d1 - self.neighborhood_size ** self.d1

    def forward(self, features,
                in_barycentric, in_lattice_offset,
                blur_neighbors,
                out_barycentric, out_lattice_offset):
        """

        :param features: float32 (B, C_in, N_in)
        :param in_barycentric: float32 (B, d1, N_in)
        :param in_lattice_offset: int64 (B, d1, N_in)
        :param blur_neighbors: int64 (B, filter_size, max_hash_cnt)
        :param out_barycentric: float32 (B, d1, N_out)
        :param out_lattice_offset: int64 (B, d1, N_out)
        :return: float32 (B, C_out, N_out) if self.sliced else (B, C_out, max_hash_cnt)
        """
        # -------------------- SLICE --------------------
        # if given lattice, batch size can only be 1 for now
        # need to add the batch effect when doing sparse sum, then minus the batch effect when slicing
        # new_lattice_offset = out_lattice_offset - (batch_indices * (max_hash_cnt + 1))[:, None, None]
        # !!! ATTENTION

        batch_size = features.size(0)
        batch_indices = torch.arange(batch_size, dtype=torch.long)
        if self.DEVICE == 'cuda':
            batch_indices = batch_indices.pin_memory()
            batch_indices = batch_indices.cuda(non_blocking=True)

        max_hash_cnt = blur_neighbors.size(-1)

        # -------------------- SPLAT --------------------
        if self.do_splat:
            # barycentric: (B, 1, d1, N_in), features: (B, feat_size, 1, N_in)
            # (B, feat_size, d1, N_in) -> (feat_size, B * d1 * N_in)
            tmp = (in_barycentric[:, None, :, :] * features[:, :, None, :]). \
                permute(1, 0, 2, 3).reshape(self.num_input, -1)
            tmp = tmp.t()  # (B * d1 * N_in, feat_size)

            # There may be -1 in blur_neighbors indicating non-existing lattice point. So need +1
            # +1 also makes the first element of splatted is 0 in all channels
            # lattice_offset: (B, d1, N_in)
            # sparse_sum: indices, values, size, cuda
            splatted = sparse_sum((in_lattice_offset + 1).reshape(1, -1), tmp,
                                  torch.Size([batch_size * (max_hash_cnt + 1), self.num_input]),
                                  self.DEVICE == 'cuda')
            splatted = splatted.reshape(batch_size, max_hash_cnt + 1, self.num_input).permute(0, 2, 1)
            # (B, feat_size, H+1)

            if self.use_norm:
                # for density normalization
                one_features = torch.ones((batch_size, 1, features.size(-1)), dtype=torch.float32)
                if self.DEVICE == 'cuda':
                    one_features = one_features.pin_memory()
                    one_features = one_features.cuda(non_blocking=True)

                # (B, d1, N_in), (B, 1=feat_size, N_in) -> (B, d1, N_in)
                one_tmp = (in_barycentric * one_features).reshape(1, -1)  # (1, B * d1 * N_in)
                one_tmp = one_tmp.t()  # (B * d1 * N_in, 1)

                one_splatted = sparse_sum((in_lattice_offset + 1).reshape(1, -1), one_tmp,
                                          torch.Size([batch_size * (max_hash_cnt + 1), 1]),
                                          self.DEVICE == 'cuda')
                one_splatted = one_splatted.reshape(batch_size, max_hash_cnt + 1)

                # print('normalize!')
                norm = 1. / (one_splatted + 1e-5)
                splatted *= norm[:, None, :]

                if DELETE_TMP_VARIABLES:
                    del one_features, one_tmp, one_splatted
        else:
            # features: (B, C, max_hash_cnt) -> (B, C, max_hash_cnt+1)
            splatted = torch.cat((torch.zeros((batch_size, self.num_input, 1),
                                              dtype=features.dtype,
                                              device=features.device),
                                  features),
                                 dim=-1)

        # -------------------- BLUR --------------------
        if self.MAX_SIZE == -1:
            chunk_size = max_hash_cnt
        else:
            chunk_size = max(1,
                             min(self.MAX_SIZE // self.num_input // self.filter_size,
                                 max_hash_cnt))
        num_chunks = (max_hash_cnt + chunk_size - 1) // chunk_size

        feat_blurred = []
        for cidx in range(num_chunks):
            start_idx = cidx * chunk_size
            end_idx = min(max_hash_cnt, start_idx + chunk_size)

            # splatted: (B, feat_size, max_hash_cnt+1)
            # blur_neighbors: (B, filter_size, max_hash_cnt), index in the range of [-1, max_hash_cnt-1]
            # spread_out: (B, feat_size, filter_size, max_hash_cnt/chunk_size)
            spread_out = splatted[batch_indices[:, None, None, None],
                                  self.feat_indices[None, :, None, None],
                                  (blur_neighbors + 1)[:, None, :, start_idx:end_idx]]
            # (B, num_input, filter_size, chunk_size)
            feat_blurred_chunk = self.blur_conv(spread_out).squeeze(2)  # (B, num_output, 1(squeezed), chunk_size)
            feat_blurred.append(feat_blurred_chunk)
        feat_blurred = torch.cat(feat_blurred, dim=-1)  # (B, num_output, max_hash_cnt)

        if not self.do_slice:
            return feat_blurred

        tmp_feat_blurred = feat_blurred[batch_indices[:, None, None, None],
                                        self.out_indices[None, :, None, None],
                                        out_lattice_offset[:, None, :, :]]
        # (B, num_output, d1, N_out)

        # barycentric: (B, d1, N_out)
        sliced = (out_barycentric[:, None, :, :] * tmp_feat_blurred).sum(dim=2)
        # (B, num_output, d1, N_out) -> (B, num_output, N_out)

        if self.use_bias:
            sliced += self.bias[None, :, None]

        return sliced
