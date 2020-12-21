import socket
import numpy as np
import yaml
import os, sys
import os.path as osp

import datasets
import models

from utils.easydict import EasyDict

model_names = sorted(name for name in models.__dict__
                     if (not name.startswith("__"))
                     and ('Args' not in name)
                     and ('Block' not in name)
                     and ('net' in name or 'Net' in name)
                     and callable(models.__dict__[name])) + ['ICP']

dataset_names = sorted(name for name in datasets.__dict__
                       if not name.startswith("__"))


def postprocess(args):
    # -------------------- miscellaneous --------------------
    args.allow_less_points = hasattr(args, 'allow_less_points') and args.allow_less_points

    # -------------------- arch --------------------
    assert (args.arch in model_names)
    # -------------------- dataset --------------------
    assert (args.dataset in dataset_names)
    assert hasattr(args, 'data_root')
    # -------------------- learning --------------------
    if not args.evaluate:
        # -------------------- init --------------------
        if not hasattr(args, 'init'):
            args.init = 'xavier'
        if not hasattr(args, 'gain'):
            args.gain = 1.

        # -------------------- custom lr --------------------
        if hasattr(args, 'custom_lr') and args.custom_lr:
            args.lrs = [float(item) for item in args.lrs.split(',')][::-1]
            args.lr_switch_epochs = [int(item) for item in args.lr_switch_epochs.split(',')][::-1]
            assert (len(args.lrs) == len(args.lr_switch_epochs))

            diffs = [second - first for first, second in zip(args.lr_switch_epochs, args.lr_switch_epochs[1:])]
            assert (np.all(np.array(diffs) < 0))

            args.lr = args.lrs[-1]

    # -------------------- resume --------------------
    if args.evaluate:
        assert (hasattr(args, 'resume'))
        assert args.resume is not False

    return args


def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as fd:
        args = yaml.load(fd, Loader=yaml.FullLoader)
        args = EasyDict(d=args)
        args = postprocess(args)
    return args

