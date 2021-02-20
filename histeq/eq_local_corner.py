import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import pyopencl as cl
import os
'''
def histEqLocalCorner(bgr, alpha=0.5, punch=0.05, blockshape=(128, 128), use_gpu=False):
    blockW = blockshape[1]
    blockH = blockshape[0]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    if not use_gpu:
        ycrcb_new = ycrcb.transpose(2, 0, 1)
        gray = ycrcb_new[0,:,:]

        mappings = np.zeros((bgr.shape[0]//blockH+1, bgr.shape[1]//blockW+1, 256), dtype=np.float32)
        histgrid = np.zeros((bgr.shape[0]//blockH+1, bgr.shape[1]//blockW+1, 256), dtype=np.float32)

        for i in range(0, bgr.shape[0]):
            for j in range(0, bgr.shape[1]):
                # soft histogramming
                b00idx = j // blockW
                b00x = b00idx * blockW
                b00idy = i // blockH
                b00y = b00idy * blockH

                s = (j - b00x) / blockW
                t = (i - b00y) / blockH

                v = gray[i, j]

                histgrid[b00idy, b00idx, v] += sqrt((1-s)**2 + (1-t)**2)
                histgrid[b00idy, b00idx+1, v] += sqrt(s**2 + (1-t)**2)
                histgrid[b00idy+1, b00idx, v] += sqrt((1-s)**2 + t**2)
                histgrid[b00idy+1, b00idx+1, v] += sqrt(s**2 + t**2)

        for i in range(0, histgrid.shape[0]):
            for j in range(0, histgrid.shape[1]):
                cdf = np.cumsum(histgrid[i, j]) / np.sum(histgrid[i, j])
                I = np.arange(0, 256)
                mappings[i,j,:] = alpha * cdf * 256 + (1 - alpha) * I

        for i in range(0, bgr.shape[0]):
            for j in range(0, bgr.shape[1]):
                # soft histogramming
                b00idx = j // blockW
                b00x = b00idx * blockW
                b00idy = i // blockH
                b00y = b00idy * blockH

                s = (j - b00x) / blockW
                t = (i - b00y) / blockH

                v = gray[i, j]

                f00 = mappings[b00idy, b00idx]
                f01 = mappings[b00idy, b00idx+1]
                f10 = mappings[b00idy+1, b00idx]
                f11 = mappings[b00idy+1, b00idx+1]
                v = (1-s) * (1-t) * f00[v] + s * (1-t) * f01[v] + (1-s) * t * f10[v] + s * t * f11[v]
                gray[i, j] = np.uint8(v)

        ycrcb = ycrcb_new.transpose(1, 2, 0)

    else: # OpenCL
        # soft histgram collection kernel

        # mapping function kernel

        # applying mapping kernel

        ycrcb = ycrcb

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
'''