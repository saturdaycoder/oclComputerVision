import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import pyopencl as cl
import os
from eq_opencl import clHistEq
from eq_global import calc_transfer_func

def histeq_local_block(gray, alpha=0.5, punch=0.05, clip=3, blockshape=(256, 256), use_gpu=True):
    blockW = blockshape[1]
    blockH = blockshape[0]

    mappings = np.zeros((gray.shape[0]//blockH, gray.shape[1]//blockW, 256), dtype=np.float32)

    if use_gpu:
        cleq = clHistEq.getInstance()
        histGrid, evt0 = cleq.histGrid(gray)
        evt0.wait()

        # TODO: opencl to merge histogram and calculate transfer func
        t1 = time.time()
        for i in range(0, gray.shape[0]//blockH):
            for j in range(0, gray.shape[1]//blockW):
                hist = histGrid[8*i, j]
                for k in range(1, 8):
                    hist += histGrid[8*i+k, j]
                mappings[i,j,:] = calc_transfer_func(hist, alpha, punch, clip).astype(np.float32)
        t2 = time.time()
    else:
        for i in range(0, gray.shape[0]//blockH):
            for j in range(0, gray.shape[1]//blockW):
                hist, _ = np.histogram(gray[i*blockH:(i+1)*blockH,j*blockW:(j+1)*blockW], bins=256, range=(0, 256))
                mappings[i,j,:] = calc_transfer_func(hist, alpha, punch, clip).astype(np.float32)

    if use_gpu:
        gray, evt2 = cleq.histeqLocalBlock(gray, mappings, blockshape)
        evt2.wait()
        histElapsed = (evt0.profile.end - evt0.profile.start)/1000000000
        mapElapsed = t2 - t1
        eqElapsed = (evt2.profile.end - evt2.profile.start)/1000000000
        print('local histogram equalization (block-based) took GPU: {:.3f} + {:.3f} ms, CPU: {:.3f} ms'.format(histElapsed*1000, eqElapsed*1000, mapElapsed*1000))
    else:
        for i in range(0, gray.shape[0]):
            for j in range(0, gray.shape[1]):
                b00idx = int((j - blockW//2) / blockW)
                b00x = b00idx * blockW + blockW//2
                b00idy = int((i - blockH//2) / blockH)
                b00y = b00idy * blockH + blockH//2
                b01idx = b00idx + 1
                b01idy = b00idy
                b10idx = b00idx
                b10idy = b00idy + 1

                if b01idx >= gray.shape[1]//blockW:
                    b01idx -= 1
                if b10idy >= gray.shape[0]//blockH:
                    b10idy -= 1

                b11idx = b01idx
                b11idy = b10idy

                s = (j - b00x) / blockW
                t = (i - b00y) / blockH

                v = gray[i, j]

                if s < 0:
                    s = 0
                if t < 0:
                    t = 0

                f00 = mappings[b00idy, b00idx]
                f01 = mappings[b01idy, b01idx]
                f10 = mappings[b10idy, b10idx]
                f11 = mappings[b11idy, b11idx]
                v1 = np.uint8((1-s) * (1-t) * f00[v] + s * (1-t) * f01[v] + (1-s) * t * f10[v] + s * t * f11[v])
                gray[i, j] = np.uint8(v1)

    return gray


