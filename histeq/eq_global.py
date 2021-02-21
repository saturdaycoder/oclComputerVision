import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
from eq_opencl import clHistEq
import os
import matplotlib.pyplot as plt
import time

def calc_transfer_func(hist, alpha, punch, clip):
    X = np.arange(0, len(hist))

    # CDF
    cdf = np.cumsum(hist) / np.sum(hist)

    # apply punch
    dark_punch = np.where(cdf >= punch)[0][0]
    bright_punch = np.where(cdf >= 1-punch)[0][0]
    hist_punched = hist[dark_punch:bright_punch]
    cdf[:dark_punch] = 0
    cdf[bright_punch:] = 1
    cdf[dark_punch:bright_punch] = np.cumsum(hist_punched) / np.sum(hist_punched)
    
    # linear blending with alpha
    # dark region
    mapping = np.zeros((256), dtype=np.float32)
    # bright region
    mapping[bright_punch:] = 255
    # blended region
    I = X
    mapping = alpha * cdf * 255 + (1-alpha) * I
    # clipping
    mapping = np.clip(mapping, 0, 255)

    # gain limiting with clip
    mapping = np.clip(mapping, I / clip, I * clip)
    return mapping

def histeq_global(bgr, alpha=1, punch=0.05, clip=2, use_gpu=True):
    cleq = clHistEq.getInstance()

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    gray = ycrcb[:,:,0].copy()

    if use_gpu:
        histGrid, evt = cleq.histGrid(gray)
        evt.wait()
        hist = histGrid.sum(axis=0).sum(axis=0)
        histElapsed = (evt.profile.end - evt.profile.start)/1000000000
    else:
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))

    t1 = time.time()
    mapping = calc_transfer_func(hist, alpha, punch, clip).astype(np.uint8)
    mapElapsed = time.time() - t1

    if use_gpu:
        gray, evt = cleq.histeqGlobal(gray, mapping)
        evt.wait()
        eqElapsed = (evt.profile.end - evt.profile.start)/1000000000
        print('GPU global histogram equalization took {:.3f} + {:.3f} + {:.3f} ms'.format(histElapsed*1000, mapElapsed*1000, eqElapsed*1000))
    else:
        eq = lambda t: mapping[t]
        vfunc = np.vectorize(eq)
        gray = vfunc(gray)

    ycrcb[:,:,0] = gray
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
