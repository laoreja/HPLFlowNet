import numpy as np
from transforms3d import euler


def evaluate_translation(t_pred, t_gt):
    error = np.linalg.norm(t_gt - t_pred, axis=-1)
    return error.mean()


def evaluate_rotation(rot_pred, rot_gt):
    if len(rot_gt.shape) == 3:
        rot_gt = rot_gt[0]
    R_pred = euler.euler2mat(rot_pred[0, 0], rot_pred[0, 1], rot_pred[0, 2])
    R_gt = euler.euler2mat(rot_gt[0, 0], rot_gt[0, 1], rot_gt[0, 2])
    return np.arccos((np.trace(np.dot(R_pred, R_gt.T)) - 1) / 2) * 180 / np.pi


def evaluate_3d(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, l2_norm


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d
