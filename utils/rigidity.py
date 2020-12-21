import torch
import numpy as np

_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_EPS4 = np.finfo(float).eps * 4.0


# TODO: Adapt for B > 1
def torch_mat2euler(M, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """

    if len(M.shape) == 2:
        M = torch.unsqueeze(M, dim=0)

    sqrt = torch.sqrt
    atan2 = torch.atan2

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if repetition:
        sy = sqrt(M[:, i, j] * M[:, i, j] + M[:, i, k] * M[:, i, k])
        if sy > _EPS4:
            ax = atan2(M[:, i, j], M[:, i, k])
            ay = atan2(sy, M[:, i, i])
            az = atan2(M[:, j, i], -M[:, k, i])
        else:
            ax = atan2(-M[:, j, k], M[:, j, j])
            ay = atan2(sy, M[:, i, i])
            az = 0.0
    else:
        cy = sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])
        if cy > _EPS4:
            ax = atan2(M[:, k, j], M[:, k, k])
            ay = atan2(-M[:, k, i], cy)
            az = atan2(M[:, j, i], M[:, i, i])
        else:
            ax = atan2(-M[:, j, k], M[:, j, j])
            ay = atan2(-M[:, k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax

    euler = torch.stack([ax, ay, az], dim=-1)
    return euler


def torch_euler2mat(euler_angles, axes='sxyz'):
    """ A gpu version euler2mat from transform3d:

    Adapted from Lv et. al Learning Rigidity

    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param euler_angles : tensor of shape (B, 3)
        First rotation angle (according to `axes`) vector of shape (B),
        Second rotation angle (according to `axes`) vector of shape (B),
        Third rotation angle (according to `axes`) vector of shape (B).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    mat : array-like shape (B, 3, 3)
    Tested w.r.t. transforms3d.euler module
    """

    ai = euler_angles[:, 0]
    aj = euler_angles[:, 1]
    ak = euler_angles[:, 2]

    cos = torch.cos
    sin = torch.sin

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj * si, sj * ci]
        c_j = [sj * sk, -cj * ss + cc, -cj * cs - sc]
        c_k = [-sj * ck, cj * sc + cs, cj * cc - ss]
    else:
        c_i = [cj * ck, sj * sc - cs, sj * cc + ss]
        c_j = [cj * sk, sj * ss + cc, sj * cs - sc]
        c_k = [-sj, cj * si, cj * ci]

    def permute(X):  # sort X w.r.t. the axis indices
        return [x for (y, x) in sorted(zip(order, X))]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r = [torch.stack(c_i, 1),
         torch.stack(c_j, 1),
         torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)
