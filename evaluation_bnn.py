import os, sys
import os.path as osp
import numpy as np
import pickle

import torch
import torch.optim
import torch.utils.data

from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

TOTAL_NUM_SAMPLES = 0


def evaluate(val_loader, model, logger, args):
    save_idx = 0
    num_sampled_batches = TOTAL_NUM_SAMPLES // args.batch_size

    # sample data for visualization
    if TOTAL_NUM_SAMPLES == 0:
        sampled_batch_indices = []
    else:
        if len(val_loader) > num_sampled_batches:
            print('num_sampled_batches', num_sampled_batches)
            print('len(val_loader)', len(val_loader))

            sep = len(val_loader) // num_sampled_batches
            sampled_batch_indices = list(range(len(val_loader)))[::sep]
        else:
            sampled_batch_indices = range(len(val_loader))

    save_dir = osp.join(args.ckpt_dir, 'visu_' + osp.split(args.ckpt_dir)[-1])
    os.makedirs(save_dir, exist_ok=True)
    path_list = []
    epe3d_list = []

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, items in enumerate(val_loader):
            pc1, pc2, sf, generated_data, path = items

            output = model(pc1, pc2, generated_data)

            pc1_np = pc1.numpy()
            pc1_np = pc1_np.transpose((0,2,1))
            pc2_np = pc2.numpy()
            pc2_np = pc2_np.transpose((0,2,1))
            sf_np = sf.numpy()
            sf_np = sf_np.transpose((0,2,1))
            output_np = output.cpu().numpy()
            output_np = output_np.transpose((0,2,1))

            EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(output_np, sf_np)

            epe3ds.update(EPE3D)
            acc3d_stricts.update(acc3d_strict)
            acc3d_relaxs.update(acc3d_relax)
            outliers.update(outlier)

            # 2D evaluation metrics
            flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
                                                            pc1_np+sf_np,
                                                            pc1_np+output_np,
                                                            path)
            EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

            epe2ds.update(EPE2D)
            acc2ds.update(acc2d)

            if i % args.print_freq == 0:
                logger.log('Test: [{0}/{1}]\t'
                           'EPE3D {epe3d_.val:.4f} ({epe3d_.avg:.4f})\t'
                           'ACC3DS {acc3d_s.val:.4f} ({acc3d_s.avg:.4f})\t'
                           'ACC3DR {acc3d_r.val:.4f} ({acc3d_r.avg:.4f})\t'
                           'Outliers3D {outlier_.val:.4f} ({outlier_.avg:.4f})\t'
                           'EPE2D {epe2d_.val:.4f} ({epe2d_.avg:.4f})\t'
                           'ACC2D {acc2d_.val:.4f} ({acc2d_.avg:.4f})'
                           .format(i + 1, len(val_loader),
                                   epe3d_=epe3ds,
                                   acc3d_s=acc3d_stricts,
                                   acc3d_r=acc3d_relaxs,
                                   outlier_=outliers,
                                   epe2d_=epe2ds,
                                   acc2d_=acc2ds,
                                   ))

            if i in sampled_batch_indices:
                np.save(osp.join(save_dir, 'pc1_' + str(save_idx) + '.npy'), pc1_np)
                np.save(osp.join(save_dir, 'sf_' + str(save_idx) + '.npy'), sf_np)
                np.save(osp.join(save_dir, 'output_' + str(save_idx) + '.npy'), output_np)
                np.save(osp.join(save_dir, 'pc2_' + str(save_idx) + '.npy'), pc2_np)
                epe3d_list.append(EPE3D)
                path_list.extend(path)
                save_idx += 1
            del pc1, pc2, sf, generated_data

    if len(path_list) > 0:
        np.save(osp.join(save_dir, 'epe3d_per_frame.npy'), np.array(epe3d_list))
        with open(osp.join(save_dir, 'sample_path_list.pickle'), 'wb') as fd:
            pickle.dump(path_list, fd)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds,
                       ))
    logger.log(res_str)
    return res_str
