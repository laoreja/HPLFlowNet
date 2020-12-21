import torch
import torch.nn as nn
from .module_utils import Conv1dReLU
from .bilateralNN import BilateralConvFlex
from .bnn_flow import BilateralCorrelationFlex


class PoseNet(nn.Module):
    def __init__(self, args):
        super(PoseNet, self).__init__()
        self.chunk_size = -1 if args.evaluate else 1024 * 1024 * 25

        self.conv_block = nn.Sequential(
            Conv1dReLU(args.dim, 32, use_leaky=args.use_leaky),
            Conv1dReLU(32, 32, use_leaky=args.use_leaky),
            Conv1dReLU(32, 64, use_leaky=args.use_leaky))

        self.bcn1 = BilateralConvFlex(args.dim, args.scales_filter_map[0][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn2 = BilateralConvFlex(args.dim, args.scales_filter_map[1][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn3 = BilateralConvFlex(args.dim, args.scales_filter_map[2][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.corr1 = BilateralCorrelationFlex(args.dim,
                                              args.scales_filter_map[2][2], args.scales_filter_map[2][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=0,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.pose_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 6))

    def forward(self, pc1, pc2, generated_data):
        feat1 = self.conv_block(pc1)
        feat2 = self.conv_block(pc2)

        pc1_out1 = self.bcn1(torch.cat((generated_data[0]['pc1_el_minus_gr'], feat1), dim=1),
                             in_barycentric=generated_data[0]['pc1_barycentric'],
                             in_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out1 = self.bcn1(torch.cat((generated_data[0]['pc2_el_minus_gr'], feat2), dim=1),
                             in_barycentric=generated_data[0]['pc2_barycentric'],
                             in_lattice_offset=generated_data[0]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[0]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc1_out2 = self.bcn2(torch.cat((generated_data[1]['pc1_el_minus_gr'], pc1_out1), dim=1),
                             in_barycentric=generated_data[1]['pc1_barycentric'],
                             in_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out2 = self.bcn2(torch.cat((generated_data[1]['pc2_el_minus_gr'], pc2_out1), dim=1),
                             in_barycentric=generated_data[1]['pc2_barycentric'],
                             in_lattice_offset=generated_data[1]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[1]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc1_out3 = self.bcn3(torch.cat((generated_data[2]['pc1_el_minus_gr'], pc1_out2), dim=1),
                             in_barycentric=generated_data[2]['pc1_barycentric'],
                             in_lattice_offset=generated_data[2]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[2]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out3 = self.bcn3(torch.cat((generated_data[2]['pc2_el_minus_gr'], pc2_out2), dim=1),
                             in_barycentric=generated_data[2]['pc2_barycentric'],
                             in_lattice_offset=generated_data[2]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[2]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        corr_out1 = self.corr1(pc1_out3, pc2_out3, prev_corr_feat=None,
                               barycentric1=None,
                               lattice_offset1=None,
                               pc1_corr_indices=generated_data[2]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[2]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[2]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[2]['pc2_hash_cnt'].item())

        pc1_avg = self.pool(pc1_out3)
        pc2_avg = self.pool(pc2_out3)
        corr_avg = self.pool(corr_out1)

        pose = self.pose_regressor(torch.cat((pc1_avg, pc2_avg, corr_avg), dim=1))

        return pose[:, :3], pose[:, 3:]
