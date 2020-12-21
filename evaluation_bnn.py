import os.path as osp
import pickle
import time

import numpy as np
import torch.optim
import torch.utils.data
from evaluation_utils import evaluate_2d, evaluate_3d, evaluate_rotation, evaluate_translation
from main_utils import *
from utils import geometry

# Sample 5 PCs during testing for visualization
TOTAL_NUM_SAMPLES = 15


def evaluate(val_loader, model, logger, args, data_gen):
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
    epe3d_histo = np.empty((1, 0))

    # 3D
    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()
    # Timing
    batch_time = AverageMeter()
    data_time = AverageMeter()
    run_time = AverageMeter()
    # 3D NR
    epe3ds_nr = AverageMeter()
    acc3d_stricts_nr = AverageMeter()
    acc3d_relaxs_nr = AverageMeter()
    outliers_nr = AverageMeter()
    # 2D NR
    epe2ds_nr = AverageMeter()
    acc2ds_nr = AverageMeter()
    # Camera Pose
    rotations = AverageMeter()
    translations = AverageMeter()

    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, items in enumerate(val_loader):
            pc1, pc2, rot_rel_gt, t_rel_gt, sf_nr_gt, sf_total_gt, generated_data, path = items

            # hack for flying things
            if sf_nr_gt.nelement() == 0:
                sf_nr_gt = torch.zeros(sf_total_gt.shape)
            if rot_rel_gt.nelement() == 0:
                rot_rel_gt = torch.zeros((1, 3))
            if t_rel_gt.nelement() == 0:
                t_rel_gt = torch.zeros((1, 3))

            # measure data loading time
            data_time.update(time.time() - start)
            start_model = time.time()
            sf_total_pred, sf_nr_pred, rot_rel_pred, t_rel_pred = model(pc1, pc2, generated_data, data_gen)
            run_time.update(time.time() - start_model)

            pc1 = pc1.numpy().transpose((0, 2, 1))
            pc2 = pc2.numpy().transpose((0, 2, 1))

            sf_total_gt = sf_total_gt.numpy().transpose((0, 2, 1))
            sf_total_pred = sf_total_pred.cpu().numpy().transpose((0, 2, 1))

            sf_nr_gt = sf_nr_gt.numpy().transpose((0, 2, 1))
            sf_nr_pred = sf_nr_pred.cpu().numpy().transpose((0, 2, 1))

            # 3D evaluation metrics
            EPE3D, acc3d_strict, acc3d_relax, outlier, l2_norm = evaluate_3d(sf_total_pred, sf_total_gt)
            epe3ds.update(EPE3D)
            acc3d_stricts.update(acc3d_strict)
            acc3d_relaxs.update(acc3d_relax)
            outliers.update(outlier)
            epe3d_histo = np.concatenate((epe3d_histo, l2_norm), axis=-1)

            # 2D evaluation metrics
            flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1,
                                                            pc1+sf_total_gt,
                                                            pc1+sf_total_pred,
                                                            path)
            EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)
            epe2ds.update(EPE2D)
            acc2ds.update(acc2d)

            # 3D
            EPE3D_nr, acc3d_strict_nr, acc3d_relax_nr, outlier_nr, _ = evaluate_3d(sf_nr_pred, sf_nr_gt)
            epe3ds_nr.update(EPE3D_nr)
            acc3d_stricts_nr.update(acc3d_strict_nr)
            acc3d_relaxs_nr.update(acc3d_relax_nr)
            outliers_nr.update(outlier_nr)
            # 2D
            flow_pred_nr, flow_gt_nr = geometry.get_batch_2d_flow(pc1,
                                                                  pc1+sf_nr_gt,
                                                                  pc1+sf_nr_pred,
                                                                  path)
            EPE2D_nr, acc2d_nr = evaluate_2d(flow_pred_nr, flow_gt_nr)
            epe2ds_nr.update(EPE2D_nr)
            acc2ds_nr.update(acc2d_nr)

            # Camera pose
            rot_rel_gt = rot_rel_gt.numpy()
            rot_rel_pred = rot_rel_pred.cpu().numpy()
            t_rel_gt = t_rel_gt.numpy()
            t_rel_pred = t_rel_pred.cpu().numpy()

            rot = evaluate_rotation(rot_rel_pred, rot_rel_gt)
            t = evaluate_translation(t_rel_pred, t_rel_gt)
            rotations.update(rot)
            translations.update(t)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

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
                np.save(osp.join(save_dir, 'pc1_' + str(save_idx) + '.npy'), pc1)
                np.save(osp.join(save_dir, 'sf_' + str(save_idx) + '.npy'), sf_total_gt)
                np.save(osp.join(save_dir, 'output_' + str(save_idx) + '.npy'), sf_total_pred)
                np.save(osp.join(save_dir, 'pc2_' + str(save_idx) + '.npy'), pc2)
                epe3d_list.append(EPE3D)
                path_list.extend(path)
                save_idx += 1
            del pc1, pc2, sf_total_gt, generated_data

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
    logger.log(' * EPE3D NR {epe3d_nr_.avg:.4f}\t'
               'ACC3DS NR {acc3d_s_nr_.avg:.4f}\t'
               'ACC3DR NR {acc3d_r_nr_.avg:.4f}\t'
               'Outliers3D NR {outlier_nr_.avg:.4f}\t'
               'EPE2D NR {epe2d_nr_.avg:.4f}\t'
               'ACC2D NR {acc2d_nr_.avg:.4f}'
               .format(
                       epe3d_nr_=epe3ds_nr,
                       acc3d_s_nr_=acc3d_stricts_nr,
                       acc3d_r_nr_=acc3d_relaxs_nr,
                       outlier_nr_=outliers_nr,
                       epe2d_nr_=epe2ds_nr,
                       acc2d_nr_=acc2ds_nr))
    logger.log(' * R {rotations_.avg:.4f}\t'
               ' t {translations_.avg:.4f}'
               .format(
                       rotations_=rotations,
                       translations_=translations))
    logger.log(
        ' * Data time {data_time_.avg:6.3f}\t'
        ' * Batch time {batch_time_.avg:6.3f}\t'
        ' * Model runtime {run_time_.avg:6.6f}\t'.format(
            data_time_=data_time,
            batch_time_=batch_time,
            run_time_=run_time)
    )

    np.save(osp.join(save_dir, 'epe3d_histo.npy'), epe3d_histo)

    return res_str
