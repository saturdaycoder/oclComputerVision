import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import pyopencl as cl
import os

def histeq_local_block(bgr, alpha=0.5, punch=0.05, clip=(0.3, 3), blockshape=(256, 256)):
    blockW = blockshape[1]
    blockH = blockshape[0]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    gray = ycrcb[:,:,0].copy()

    mappings = np.zeros((bgr.shape[0]//blockH+1, bgr.shape[1]//blockW+1, 256), dtype=np.float64)
    for i in range(0, bgr.shape[0]//blockH):
        for j in range(0, bgr.shape[1]//blockW):
            hist, _ = np.histogram(gray[i*blockH:(i+1)*blockH,j*blockW:(j+1)*blockW], bins=256, range=(0, 256))
            cdf = np.cumsum(hist) / np.sum(hist)
            I = np.arange(0, len(hist))
            mappings[i,j,:] = np.clip(alpha * cdf * len(hist) + (1 - alpha) * I, I * clip[0], I * clip[1])

    for i in range(0, bgr.shape[0]):
        for j in range(0, bgr.shape[1]):
            b00idx = (j - blockW//2) // blockW
            b00x = b00idx * blockW + blockW//2
            b00idy = (i - blockH//2) // blockH
            b00y = b00idy * blockH + blockH//2

            s = (j - b00x) / blockW
            t = (i - b00y) / blockH

            v = gray[i, j]

            if s < 0:
                s = 0
            elif s > 1:
                s = 1
            if t < 0:
                t = 0
            elif t > 1:
                t = 1

            f00 = mappings[b00idy, b00idx]
            f01 = mappings[b00idy, b00idx+1]
            f10 = mappings[b00idy+1, b00idx]
            f11 = mappings[b00idy+1, b00idx+1]
            v1 = np.uint8((1-s) * (1-t) * f00[v] + s * (1-t) * f01[v] + (1-s) * t * f10[v] + s * t * f11[v])
            #v1 = np.clip(v1, v * clip[0], v * clip[1])
            gray[i, j] = np.uint8(v1)

    ycrcb[:,:,0] = gray
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


