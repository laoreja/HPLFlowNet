import os.path as osp
import time
from functools import partial
import gc
import traceback

import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import transforms
import datasets
import models
import cmd_args
from main_utils import *

from models import EPE3DLoss
from evaluation_bnn import evaluate

from torch.utils import tensorboard
import models.signals as ss


def main():
    # ensure numba JIT is on
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    # parse arguments
    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    # -------------------- logging args --------------------
    if osp.exists(args.ckpt_dir):
        to_continue = query_yes_no('Attention!!!, ckpt_dir already exists!\
                                        Whether to continue?',
                                   default=None)
        if not to_continue:
            sys.exit(1)
    os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)

    logger = Logger(osp.join(args.ckpt_dir, 'log'))
    logger.log('sys.argv:\n' + ' '.join(sys.argv))

    os.environ['NUMBA_NUM_THREADS'] = str(args.workers)
    logger.log('NUMBA NUM THREADS\t' + os.environ['NUMBA_NUM_THREADS'])

    for arg in sorted(vars(args)):
        logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
    logger.log('')

    data_gen = transforms.GenerateDataUnsymmetric(args)
    # -------------------- dataset & loader --------------------
    if not args.evaluate:
        train_dataset = datasets.__dict__[args.dataset](
            train=True,
            transform=transforms.Augmentation(args.aug_together,
                                              args.aug_pc2,
                                              args.data_process,
                                              args.num_points,
                                              args.allow_less_points),
            gen_func=data_gen,
            args=args
        )
        logger.log('train_dataset: ' + str(train_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
            drop_last=True
        )
        # --------------- Set up tensor board ------------------
        tf_board_writer = tensorboard.SummaryWriter(args.tf_board_dir)

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        gen_func=data_gen,
        args=args
    )
    logger.log('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
        drop_last=True
    )

    # -------------------- create model --------------------
    logger.log("=>  creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](args)

    if not args.evaluate:
        init_func = partial(init_weights_multi, init_type=args.init, gain=args.gain)
        model.apply(init_func)
    logger.log(model)

    logger.log('Detected GPUs: ' + str(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).to(args.DEVICE)
    criterion = EPE3DLoss().to(args.DEVICE)

    if args.evaluate:
        torch.backends.cudnn.enabled = False
    else:
        cudnn.benchmark = True
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # But if your input sizes changes at each iteration,
    # then cudnn will benchmark every time a new size appears,
    # possibly leading to worse runtime performances.

    # -------------------- resume --------------------
    if args.resume:
        if osp.isfile(args.resume):
            logger.log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            logger.log("=> loaded checkpoint '{}' (start epoch {}, min loss {})"
                       .format(args.resume, checkpoint['epoch'], checkpoint['min_loss']))
        else:
            logger.log("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = None
    else:
        args.start_epoch = 0

    # -------------------- evaluation --------------------
    if args.evaluate:
        res_str = evaluate(val_loader, model, logger, args, data_gen)
        logger.close()
        return res_str

    # -------------------- optimizer --------------------
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=0)
    if args.resume and (checkpoint is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])

    if hasattr(args, 'reset_lr') and args.reset_lr:
        print('reset lr')
        reset_learning_rate(optimizer, args)

    # -------------------- main loop --------------------
    min_train_loss = None
    best_train_epoch = None
    best_val_epoch = None
    do_eval = True

    # init loss_weights.fb on resume
    if args.fb_weight_scale:
        args.loss_weights.fb = init_weight_fb(args.fb_scale_beg, args.fb_scale_end, args.fb_scale_step, args)

    for epoch in range(args.start_epoch, args.epochs):
        old_lr = optimizer.param_groups[0]['lr']
        adjust_learning_rate(optimizer, epoch, args)

        lr = optimizer.param_groups[0]['lr']
        if old_lr != lr:
            print('Switch lr!')
        logger.log('lr: ' + str(optimizer.param_groups[0]['lr']))

        # increase FB weight
        if args.fb_weight_scale:
            args.loss_weights.fb = adjust_weight_fb(epoch, args.loss_weights.fb,
                                                    args.fb_scale_beg, args.fb_scale_end, args.fb_scale_step)
            logger.log('fb_loss_weight: {:.2f}'.format(args.loss_weights.fb))

        train_loss = train(train_loader, model, criterion, optimizer, epoch, logger, tf_board_writer, data_gen)
        gc.collect()

        is_train_best = True if best_train_epoch is None else (train_loss < min_train_loss)
        if is_train_best:
            min_train_loss = train_loss
            best_train_epoch = epoch

        if do_eval:
            val_loss = validate(val_loader, model, criterion, epoch, len(train_loader.dataset), logger,
                                tf_board_writer, data_gen)
            gc.collect()

            is_val_best = True if best_val_epoch is None else (val_loss < min_val_loss)
            if is_val_best:
                min_val_loss = val_loss
                best_val_epoch = epoch
                logger.log("New min val loss!")

        min_loss = min_val_loss if do_eval else min_train_loss
        is_best = is_val_best if do_eval else is_train_best
        save_checkpoint({
            'epoch': epoch + 1,  # next start epoch
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.ckpt_dir)

    train_str = 'Best train loss: {:.5f} at epoch {:3d}'.format(min_train_loss, best_train_epoch)
    logger.log(train_str)

    if do_eval:
        val_str = 'Best val loss: {:.5f} at epoch {:3d}'.format(min_val_loss, best_val_epoch)
        logger.log(val_str)

    logger.close()
    result_str = val_str if do_eval else train_str
    return result_str


def train(train_loader, model, criterion, optimizer, epoch, logger, tf_board_writer, data_gen):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    epe3d_losses = AverageMeter()
    epe3d_nr_losses = AverageMeter()
    rot_losses = AverageMeter()
    t_losses = AverageMeter()
    fb_cycle_losses = AverageMeter()
    nn_losses = AverageMeter()

    model.train()

    w = args.loss_weights

    start = time.time()
    for batch_idx, (pc1, pc2, rot_gt, t_gt, sf_nr_gt, sf_total_gt, generated_data, path) in enumerate(train_loader):
        try:
            # measure data loading time
            data_time.update(time.time() - start)
            batch_size = len(pc1)
            # forward pass
            pc1 = pc1.to(args.DEVICE, non_blocking=True)
            pc2 = pc2.to(args.DEVICE, non_blocking=True)
            sf_total_pred, sf_nr_pred, rot_pred, t_pred = model(pc1, pc2, generated_data, data_gen)

            # supervised part
            if sf_total_gt.nelement() != 0:
                sf_total_gt = sf_total_gt.to(args.DEVICE, non_blocking=True)
                sf_nr_gt = sf_nr_gt.to(args.DEVICE, non_blocking=True)
                rot_gt = rot_gt.to(args.DEVICE, non_blocking=True)
                t_gt = t_gt.to(args.DEVICE, non_blocking=True)
                # Compute losses
                rot_loss = criterion(input=rot_pred, target=rot_gt).mean()
                t_loss = criterion(input=t_pred, target=t_gt).mean()
                epe3d_nr_loss = criterion(input=sf_nr_pred, target=sf_nr_gt).mean()
                epe3d_loss = criterion(input=sf_total_pred, target=sf_total_gt).mean()
            else:
                epe3d_loss = torch.tensor(0, device=args.DEVICE)
                epe3d_nr_loss = torch.tensor(0, device=args.DEVICE)
                rot_loss = torch.tensor(0, device=args.DEVICE)
                t_loss = torch.tensor(0, device=args.DEVICE)

            # self-supervision signals
            if w.nn != 0 and w.fb != 0:
                pc2_transformed, nn_pc2_transformed, pc1_cycle = ss.fb_consistency(model, pc1, pc2, sf_total_pred,
                                                                                   data_gen)
                nn_loss = criterion(input=pc2_transformed, target=nn_pc2_transformed).mean()
                fb_cycle_loss = criterion(input=pc1_cycle, target=pc1).mean()
            else:
                fb_cycle_loss = torch.tensor(0, device=args.DEVICE)
                nn_loss = torch.tensor(0, device=args.DEVICE)

            # compute total loss
            camera_pose_loss = w.rotation * rot_loss + w.translation * t_loss
            ss_loss = w.fb * fb_cycle_loss + w.nn * nn_loss
            total_loss = w.epe3d * epe3d_loss + w.epe3d_nr * epe3d_nr_loss + camera_pose_loss + ss_loss

            # Record loss
            epe3d_losses.update(epe3d_loss.item(), batch_size)
            epe3d_nr_losses.update(epe3d_nr_loss.item(), batch_size)
            rot_losses.update(rot_loss.item(), batch_size)
            t_losses.update(t_loss.item(), batch_size)
            fb_cycle_losses.update(fb_cycle_loss.item(), batch_size)
            nn_losses.update(nn_loss.item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)

            # Gradients, SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            # Intermediate logging of metrics
            if batch_idx % args.print_freq == 0:
                logger.log('Epoch: [{0}][{1}/{2}]\t'
                           'Total Loss {total_losses_.val:.4f} ({total_losses_.avg:.4f})\t'
                           'EPE3D Loss {epe3d_losses_.val:.4f} ({epe3d_losses_.avg:.4f})\t'
                           'EPE3D NR Loss {epe3d_nr_losses_.val:.4f} ({epe3d_nr_losses_.avg:.4f})\t'
                           'Rot Loss {rot_losses_.val:.4f} ({rot_losses_.avg:.4f})\t'
                           't Loss {t_losses_.val:.4f} ({t_losses_.avg:.4f})\t'
                           'FB Loss {fb_cycle_losses_.val:.4f} ({fb_cycle_losses_.avg:.4f})\t'
                           'NN Loss {nn_losses_.val:.4f} ({nn_losses_.avg:.4f})'.format(
                                epoch + 1, batch_size * batch_idx + 1, len(train_loader.dataset),
                                total_losses_=total_losses,
                                epe3d_losses_=epe3d_losses,
                                epe3d_nr_losses_=epe3d_nr_losses,
                                rot_losses_=rot_losses,
                                t_losses_=t_losses,
                                fb_cycle_losses_=fb_cycle_losses,
                                nn_losses_=nn_losses), end='')
                logger.log('')
                # Plot the loss function
                global_step = epoch * len(train_loader.dataset) + batch_size * batch_idx
                tf_board_writer.add_scalar('Total_loss/train', total_losses.avg, global_step)
                tf_board_writer.add_scalar('EPE3D_loss/train', epe3d_losses.avg, global_step)
                tf_board_writer.add_scalar('EPE3D_NR_loss/train', epe3d_nr_losses.avg, global_step)
                tf_board_writer.add_scalar('Rotation_loss/train', rot_losses.avg, global_step)
                tf_board_writer.add_scalar('Translation_loss/train', t_losses.avg, global_step)
                tf_board_writer.add_scalar('FB_loss/train', fb_cycle_losses.avg, global_step)
                tf_board_writer.add_scalar('NN_loss/train', nn_losses.avg, global_step)
        except RuntimeError as ex:
            logger.log("in TRAIN, RuntimeError " + repr(ex))
            logger.log("batch idx: " + str(batch_idx) + ' path: ' + path[0])
            traceback.print_tb(ex.__traceback__, file=logger.out_fd)
            traceback.print_tb(ex.__traceback__)

            if "CUDA out of memory" in str(ex):
                logger.log(" * Releasing memory. Continuing training")
                logger.log(" * Total loss = " + str(total_losses.val))

                del pc1, pc2, rot_gt, t_gt, sf_nr_gt, sf_total_gt
                if 'epe3d_loss' in locals():
                    del epe3d_loss
                if 'fb_cycle_loss' in locals():
                    del fb_cycle_loss
                if 'nn_loss' in locals():
                    del nn_loss
                if 'total_loss' in locals():
                    del total_loss
                if 'sf_total_pred' in locals():
                    del sf_total_pred
                if 'sf_nr_pred' in locals():
                    del sf_nr_pred
                if 'rot_pred' in locals():
                    del rot_pred
                if 't_pred' in locals():
                    del t_pred
                if 'pc2_transformed' in locals():
                    del pc2_transformed
                if 'nn_pc2_transformed' in locals():
                    del nn_pc2_transformed
                if 'pc1_cycle' in locals():
                    del pc1_cycle

                torch.cuda.empty_cache()
                gc.collect()
            elif "merge_sort: failed to synchronize" in str(ex):
                logger.log(" * Continuing training")
                logger.log(" * Total loss = " + str(total_losses.val))
            else:
                logger.log(" * Cannot continue. Program exit")
                sys.exit(1)

    # Print epoch results
    logger.log(
        ' * Train Total {total_losses_.avg:.4f}\t'
        ' * Train EPE3D {epe3d_losses_.avg:.4f}\t'
        ' * Train EPE3D NR {epe3d_nr_losses_.avg:.4f}\t'
        ' * Train Rot {rot_losses_.avg:.4f}\t'
        ' * Train t {t_losses_.avg:.4f}\t'
        ' * Train FB {fb_cycle_losses_.avg:.4f}\t'
        ' * Train NN {nn_losses_.avg:.4f}'.format(
            total_losses_=total_losses,
            epe3d_losses_=epe3d_losses,
            epe3d_nr_losses_=epe3d_nr_losses,
            rot_losses_=rot_losses,
            t_losses_=t_losses,
            fb_cycle_losses_=fb_cycle_losses,
            nn_losses_=nn_losses))
    logger.log(
        ' * Train Data time {data_time_.avg:6.3f}\t'
        ' * Train Batch time {batch_time_.avg:6.3f}'.format(
            data_time_=data_time,
            batch_time_=batch_time))
    return total_losses.avg


def validate(val_loader, model, criterion, epoch, len_train, logger, tf_board_writer, data_gen):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epe3d_losses = AverageMeter()
    epe3d_nr_losses = AverageMeter()
    rot_losses = AverageMeter()
    t_losses = AverageMeter()
    fb_cycle_losses = AverageMeter()
    nn_losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        start = time.time()
        for batch_idx, (pc1, pc2, rot_gt, t_gt, sf_nr_gt, sf_total_gt, generated_data, _) in enumerate(val_loader):
            try:
                # measure data loading time
                data_time.update(time.time() - start)

                # forward pass
                pc1 = pc1.to(args.DEVICE, non_blocking=True)
                pc2 = pc2.to(args.DEVICE, non_blocking=True)
                sf_total_pred, sf_nr_pred, rot_pred, t_pred = model(pc1, pc2, generated_data, data_gen)

                if sf_total_gt.nelement() != 0:
                    sf_total_gt = sf_total_gt.to(args.DEVICE, non_blocking=True)
                    sf_nr_gt = sf_nr_gt.to(args.DEVICE, non_blocking=True)
                    rot_gt = rot_gt.to(args.DEVICE, non_blocking=True)
                    t_gt = t_gt.to(args.DEVICE, non_blocking=True)
                    epe3d_loss = criterion(input=sf_total_pred, target=sf_total_gt).mean()
                    epe3d_nr_loss = criterion(input=sf_nr_pred, target=sf_nr_gt).mean()
                    rot_loss = torch.norm(rot_pred - rot_gt, p=2, dim=1).mean()
                    t_loss = torch.norm(t_pred - t_gt, p=2, dim=1).mean()
                else:
                    epe3d_loss = torch.tensor(0, device=args.DEVICE)
                    epe3d_nr_loss = torch.tensor(0, device=args.DEVICE)
                    rot_loss = torch.tensor(0, device=args.DEVICE)
                    t_loss = torch.tensor(0, device=args.DEVICE)

                if args.loss_weights.nn != 0 and args.loss_weights.fb != 0:
                    pc2_transformed, nn_pc2_transformed, pc1_cycle = ss.fb_consistency(model, pc1, pc2, sf_total_pred,
                                                                                       data_gen)
                    nn_loss = criterion(input=pc2_transformed, target=nn_pc2_transformed).mean()
                    fb_cycle_loss = criterion(input=pc1_cycle, target=pc1).mean()
                else:
                    fb_cycle_loss = torch.tensor(0, device=args.DEVICE)
                    nn_loss = torch.tensor(0, device=args.DEVICE)

                epe3d_losses.update(epe3d_loss.item(), len(pc1))
                epe3d_nr_losses.update(epe3d_nr_loss.item(), len(pc1))
                rot_losses.update(rot_loss.item(), len(pc1))
                t_losses.update(t_loss.item(), len(pc1))
                fb_cycle_losses.update(fb_cycle_loss.item(), len(pc1))
                nn_losses.update(nn_loss.item(), len(pc1))

                # measure elapsed time
                batch_time.update(time.time() - start)
                start = time.time()

                if batch_idx % args.print_freq == 0:
                    logger.log('Test: [{0}/{1}]\t'
                               'EPE3D loss {epe3d_losses_.val:.4f} ({epe3d_losses_.avg:.4f})\t'
                               'EPE3D NR Loss {epe3d_nr_losses_.val:.4f} ({epe3d_nr_losses_.avg:.4f})\t'
                               'Rot Loss {rot_losses_.val:.4f} ({rot_losses_.avg:.4f})\t'
                               't Loss {t_losses_.val:.4f} ({t_losses_.avg:.4f})\t'
                               'FB Loss {fb_cycle_losses_.val:.4f} ({fb_cycle_losses_.avg:.4f})\t'
                               'NN Loss {nn_losses_.val:.4f} ({nn_losses_.avg:.4f})'.format(
                                    len(pc1) * batch_idx + 1, len(val_loader.dataset),
                                    epe3d_losses_=epe3d_losses,
                                    epe3d_nr_losses_=epe3d_nr_losses,
                                    rot_losses_=rot_losses,
                                    t_losses_=t_losses,
                                    fb_cycle_losses_=fb_cycle_losses,
                                    nn_losses_=nn_losses))
            except RuntimeError as ex:
                logger.log("in VAL, RuntimeError " + repr(ex))
                traceback.print_tb(ex.__traceback__, file=logger.out_fd)
                traceback.print_tb(ex.__traceback__)

                if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    logger.log("out of memory, continue")
                    del pc1, pc2, rot_gt, t_gt, sf_nr_gt, sf_total_gt
                    torch.cuda.empty_cache()
                    gc.collect()
                    print('remained objects after OOM crash')
                else:
                    sys.exit(1)

    logger.log(' * EPE3D loss {epe3d_loss_.avg:.4f}\t'
               ' * EPE3D NR {epe3d_nr_losses_.avg:.4f}\t'
               ' * Rot {rot_losses_.avg:.4f}\t'
               ' * t {t_losses_.avg:.4f}\t'
               ' * FB {fb_cycle_losses_.avg:.4f}\t'
               ' * NN {nn_losses_.avg:.4f}'.format(
                    epe3d_loss_=epe3d_losses,
                    epe3d_nr_losses_=epe3d_nr_losses,
                    rot_losses_=rot_losses,
                    t_losses_=t_losses,
                    fb_cycle_losses_=fb_cycle_losses,
                    nn_losses_=nn_losses))
    logger.log(
        ' * Test Data time {data_time_.avg:6.3f}\t'
        ' * Test Batch time {batch_time_.avg:6.3f}'.format(
            data_time_=data_time,
            batch_time_=batch_time))
    # Plot the loss function
    tf_board_writer.add_scalar('EPE3D_loss/validation', epe3d_losses.avg, (epoch + 1) * len_train)
    tf_board_writer.add_scalar('EPE3D_NR_loss/validation', epe3d_nr_losses.avg, (epoch + 1) * len_train)
    tf_board_writer.add_scalar('Rotation_loss/validation', rot_losses.avg, (epoch + 1) * len_train)
    tf_board_writer.add_scalar('Translation_loss/validation', t_losses.avg, (epoch + 1) * len_train)
    tf_board_writer.add_scalar('FB_loss/validation', fb_cycle_losses.avg, (epoch + 1) * len_train)
    tf_board_writer.add_scalar('NN_loss/validation', nn_losses.avg, (epoch + 1) * len_train)

    # if there is no gt use self-supervised part as proxy
    gt_available = sf_total_gt.nelement() != 0 and args.loss_weights.epe3d != 0
    return epe3d_losses.avg if gt_available else fb_cycle_losses.avg + nn_losses.avg


if __name__ == '__main__':
    main()
