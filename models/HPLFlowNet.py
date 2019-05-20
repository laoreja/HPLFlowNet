import torch
import torch.nn as nn

from .bilateralNN import BilateralConvFlex
from .bnn_flow import BilateralCorrelationFlex
from .module_utils import Conv1dReLU

__all__ = ['HPLFlowNet']


class HPLFlowNet(nn.Module):
    def __init__(self, args):
        super(HPLFlowNet, self).__init__()
        self.scales_filter_map = args.scales_filter_map
        assert len(self.scales_filter_map) == 7

        self.chunk_size = -1 if args.evaluate else 1024 * 1024 * 25

        conv_module = Conv1dReLU

        self.conv1 = nn.Sequential(
            conv_module(args.dim, 32, use_leaky=args.use_leaky),
            conv_module(32, 32, use_leaky=args.use_leaky),
            conv_module(32, 64, use_leaky=args.use_leaky), )

        self.bcn1 = BilateralConvFlex(args.dim, self.scales_filter_map[0][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn1_ = BilateralConvFlex(args.dim, self.scales_filter_map[0][1],
                                       args.dim + 1 + 64 + 512, [1024, 1024],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.bcn2 = BilateralConvFlex(args.dim, self.scales_filter_map[1][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn2_ = BilateralConvFlex(args.dim, self.scales_filter_map[1][1],
                                       args.dim + 1 + 64 + 256, [512, 512],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.bcn3 = BilateralConvFlex(args.dim, self.scales_filter_map[2][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn3_ = BilateralConvFlex(args.dim, self.scales_filter_map[2][1],
                                       args.dim + 1 + 64 * 2 + 256, [256, 256],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.corr1 = BilateralCorrelationFlex(args.dim,
                                              self.scales_filter_map[2][2], self.scales_filter_map[2][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=0,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.bcn4 = BilateralConvFlex(args.dim, self.scales_filter_map[3][1],
                                      64 + args.dim + 1, [64, 64], args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn4_ = BilateralConvFlex(args.dim, self.scales_filter_map[3][1],
                                       args.dim + 1 + 64 * 2 + 128, [256, 256],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.corr2 = BilateralCorrelationFlex(args.dim,
                                              self.scales_filter_map[3][2], self.scales_filter_map[3][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=64,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.bcn5 = BilateralConvFlex(args.dim, self.scales_filter_map[4][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn5_ = BilateralConvFlex(args.dim, self.scales_filter_map[4][1],
                                       args.dim + 1 + 64 * 2 + 128, [128, 128],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.corr3 = BilateralCorrelationFlex(args.dim,
                                              self.scales_filter_map[4][2], self.scales_filter_map[4][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=64,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.bcn6 = BilateralConvFlex(args.dim, self.scales_filter_map[5][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn6_ = BilateralConvFlex(args.dim, self.scales_filter_map[5][1],
                                       args.dim + 1 + 64 * 2 + 128, [128, 128],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.corr4 = BilateralCorrelationFlex(args.dim,
                                              self.scales_filter_map[5][2], self.scales_filter_map[5][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=64,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.bcn7 = BilateralConvFlex(args.dim, self.scales_filter_map[6][1],
                                      64 + args.dim + 1, [64, 64],
                                      args.DEVICE,
                                      use_bias=args.bcn_use_bias,
                                      use_leaky=args.use_leaky,
                                      use_norm=args.bcn_use_norm,
                                      do_splat=True,
                                      do_slice=False,
                                      last_relu=args.last_relu,
                                      chunk_size=self.chunk_size)

        self.bcn7_ = BilateralConvFlex(args.dim, self.scales_filter_map[6][1],
                                       64 * 2, [128, 128],
                                       args.DEVICE,
                                       use_bias=args.bcn_use_bias,
                                       use_leaky=args.use_leaky,
                                       use_norm=args.bcn_use_norm,
                                       do_splat=False,
                                       do_slice=True,
                                       last_relu=args.last_relu,
                                       chunk_size=self.chunk_size)

        self.corr5 = BilateralCorrelationFlex(args.dim,
                                              self.scales_filter_map[6][2], self.scales_filter_map[6][3],
                                              64, [32, 32], [64, 64],
                                              args.DEVICE,
                                              use_bias=args.bcn_use_bias,
                                              use_leaky=args.use_leaky,
                                              use_norm=args.bcn_use_norm,
                                              prev_corr_dim=64,
                                              last_relu=args.last_relu,
                                              chunk_size=self.chunk_size)

        self.conv2 = conv_module(1024, 1024, use_leaky=args.use_leaky)
        self.conv3 = conv_module(1024, 512, use_leaky=args.use_leaky)
        self.conv4 = nn.Conv1d(512, 3, kernel_size=1)

    def forward(self, pc1, pc2, generated_data):
        feat1 = self.conv1(pc1)
        feat2 = self.conv1(pc2)

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
                               max_hash_cnt2=generated_data[2]['pc2_hash_cnt'].item(),
                               )

        pc1_out4 = self.bcn4(torch.cat((generated_data[3]['pc1_el_minus_gr'], pc1_out3), dim=1),
                             in_barycentric=generated_data[3]['pc1_barycentric'],
                             in_lattice_offset=generated_data[3]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[3]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out4 = self.bcn4(torch.cat((generated_data[3]['pc2_el_minus_gr'], pc2_out3), dim=1),
                             in_barycentric=generated_data[3]['pc2_barycentric'],
                             in_lattice_offset=generated_data[3]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[3]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        corr_out2 = self.corr2(pc1_out4, pc2_out4, corr_out1,
                               barycentric1=generated_data[3]['pc1_barycentric'],
                               lattice_offset1=generated_data[3]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[3]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[3]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[3]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[3]['pc2_hash_cnt'].item(),
                               )

        pc1_out5 = self.bcn5(torch.cat((generated_data[4]['pc1_el_minus_gr'], pc1_out4), dim=1),
                             in_barycentric=generated_data[4]['pc1_barycentric'],
                             in_lattice_offset=generated_data[4]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[4]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out5 = self.bcn5(torch.cat((generated_data[4]['pc2_el_minus_gr'], pc2_out4), dim=1),
                             in_barycentric=generated_data[4]['pc2_barycentric'],
                             in_lattice_offset=generated_data[4]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[4]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        corr_out3 = self.corr3(pc1_out5, pc2_out5, corr_out2,
                               barycentric1=generated_data[4]['pc1_barycentric'],
                               lattice_offset1=generated_data[4]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[4]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[4]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[4]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[4]['pc2_hash_cnt'].item(),
                               )

        pc1_out6 = self.bcn6(torch.cat((generated_data[5]['pc1_el_minus_gr'], pc1_out5), dim=1),
                             in_barycentric=generated_data[5]['pc1_barycentric'],
                             in_lattice_offset=generated_data[5]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[5]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out6 = self.bcn6(torch.cat((generated_data[5]['pc2_el_minus_gr'], pc2_out5), dim=1),
                             in_barycentric=generated_data[5]['pc2_barycentric'],
                             in_lattice_offset=generated_data[5]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[5]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        corr_out4 = self.corr4(pc1_out6, pc2_out6, corr_out3,
                               barycentric1=generated_data[5]['pc1_barycentric'],
                               lattice_offset1=generated_data[5]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[5]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[5]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[5]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[5]['pc2_hash_cnt'].item(),
                               )

        pc1_out7 = self.bcn7(torch.cat((generated_data[6]['pc1_el_minus_gr'], pc1_out6), dim=1),
                             in_barycentric=generated_data[6]['pc1_barycentric'],
                             in_lattice_offset=generated_data[6]['pc1_lattice_offset'],
                             blur_neighbors=generated_data[6]['pc1_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        pc2_out7 = self.bcn7(torch.cat((generated_data[6]['pc2_el_minus_gr'], pc2_out6), dim=1),
                             in_barycentric=generated_data[6]['pc2_barycentric'],
                             in_lattice_offset=generated_data[6]['pc2_lattice_offset'],
                             blur_neighbors=generated_data[6]['pc2_blur_neighbors'],
                             out_barycentric=None, out_lattice_offset=None)

        corr_out5 = self.corr5(pc1_out7, pc2_out7, corr_out4,
                               barycentric1=generated_data[6]['pc1_barycentric'],
                               lattice_offset1=generated_data[6]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[6]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[6]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[6]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[6]['pc2_hash_cnt'].item(),
                               )

        # upsample
        pc1_out7_back = self.bcn7_(torch.cat((corr_out5, pc1_out7), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[6]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[6]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[6]['pc1_lattice_offset'],
                                   )

        pc1_out6_back = self.bcn6_(
            torch.cat((generated_data[6]['pc1_el_minus_gr'], pc1_out7_back, corr_out4, pc1_out6), dim=1),
            in_barycentric=None, in_lattice_offset=None,
            blur_neighbors=generated_data[5]['pc1_blur_neighbors'],
            out_barycentric=generated_data[5]['pc1_barycentric'],
            out_lattice_offset=generated_data[5]['pc1_lattice_offset'],
            )

        pc1_out5_back = self.bcn5_(
            torch.cat((generated_data[5]['pc1_el_minus_gr'], pc1_out6_back, corr_out3, pc1_out5), dim=1),
            in_barycentric=None, in_lattice_offset=None,
            blur_neighbors=generated_data[4]['pc1_blur_neighbors'],
            out_barycentric=generated_data[4]['pc1_barycentric'],
            out_lattice_offset=generated_data[4]['pc1_lattice_offset'],
            )

        pc1_out4_back = self.bcn4_(
            torch.cat((generated_data[4]['pc1_el_minus_gr'], pc1_out5_back, corr_out2, pc1_out4), dim=1),
            in_barycentric=None, in_lattice_offset=None,
            blur_neighbors=generated_data[3]['pc1_blur_neighbors'],
            out_barycentric=generated_data[3]['pc1_barycentric'],
            out_lattice_offset=generated_data[3]['pc1_lattice_offset'],
            )

        pc1_out3_back = self.bcn3_(
            torch.cat((generated_data[3]['pc1_el_minus_gr'], pc1_out4_back, corr_out1, pc1_out3), dim=1),
            in_barycentric=None, in_lattice_offset=None,
            blur_neighbors=generated_data[2]['pc1_blur_neighbors'],
            out_barycentric=generated_data[2]['pc1_barycentric'],
            out_lattice_offset=generated_data[2]['pc1_lattice_offset'],
            )

        pc1_out2_back = self.bcn2_(torch.cat((generated_data[2]['pc1_el_minus_gr'], pc1_out3_back, pc1_out2), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[1]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                                   )

        pc1_out1_back = self.bcn1_(torch.cat((generated_data[1]['pc1_el_minus_gr'], pc1_out2_back, pc1_out1), dim=1),
                                   in_barycentric=None, in_lattice_offset=None,
                                   blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                                   out_barycentric=generated_data[0]['pc1_barycentric'],
                                   out_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                                   )

        # FINAL
        res = self.conv2(pc1_out1_back)
        res = self.conv3(res)
        res = self.conv4(res)

        return res
