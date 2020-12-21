"""
MIT License

Copyright (c) 2018 Zhaoyang Lv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import re
import imageio
from PIL import Image, ImageDraw, ImageFont

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()


def pngdepth_read(filename):
    return imageio.imread(filename).astype(np.float32) / 1000.0


def png_invalid_read(filename):
    return imageio.imread(filename)


def readPFM(file):
    """ read the file in pfm format
    Original code from the Scene Flow Dataset (CVPR 2016), Freiburg
    """

    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def flow_visualize(flow, max_range=1e3):
    """ Original code from SINTEL toolbox, by Jonas Wulff.
    """
    import matplotlib.colors as colors
    du = flow[:, :, 0]
    dv = flow[:, :, 1]
    [h, w] = du.shape
    max_flow = min(max_range, np.max(np.sqrt(du * du + dv * dv)))
    img = np.ones((h, w, 3), dtype=np.float64)
    # angle layer
    img[:, :, 0] = (np.arctan2(dv, du) / (2 * np.pi) + 1) % 1.0
    # magnitude layer, normalized to 1
    img[:, :, 1] = np.sqrt(du * du + dv * dv) / (max_flow + 1e-8)
    # phase layer
    # img[:, :, 2] = valid
    # convert to rgb
    img = colors.hsv_to_rgb(img)
    # remove invalid point
    img[:, :, 0] = img[:, :, 0]
    img[:, :, 1] = img[:, :, 1]
    img[:, :, 2] = img[:, :, 2]
    return img


def flow_read_from_flo(filename):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    tmp = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width * 2))
    u = tmp[:, np.arange(width) * 2]
    v = tmp[:, np.arange(width) * 2 + 1]
    return u, v

