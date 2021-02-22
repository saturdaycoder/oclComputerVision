import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import os
from eq_opencl import clHistEq
from eq_global import calc_transfer_func

def histeq_local_corner(gray, alpha=0.5, punch=0.05, clip=3, blockshape=(256, 256), use_gpu=True):
    blockH = blockshape[0]
    blockW = blockshape[1]

    mappings = np.zeros((gray.shape[0]//blockH+1, gray.shape[1]//blockW+1, 256), dtype=np.float32)
    histgrid = np.zeros((gray.shape[0]//blockH+1, gray.shape[1]//blockW+1, 256), dtype=np.float32)

    if use_gpu:
        return gray
        cleq = clHistEq.getInstance()

        # soft histogramming
        histGridCl, evt0 = cleq.histGridSoftWeighted(gray, blockshape)
        evt0.wait()

        print('soft hist took {:.3f} ms'.format((evt0.profile.end - evt0.profile.start) / 1000000))

        # merge histogram pieces
        for i in range(0, histgrid.shape[0]-1):
            for j in range(0, histgrid.shape[1]-1):
                for k in range(0, 16):
                    # LT
                    histgrid[i, j] += histGridCl[i*16+k, j, 0]
                    # RT
                    histgrid[i, j+1] += histGridCl[i*16+k, j, 1]
                    # LB
                    histgrid[i+1, j] += histGridCl[i*16+k, j, 2]
                    # RB
                    histgrid[i+1, j+1] += histGridCl[i*16+k, j, 3]
        
        # TODO: opencl to calculate transfer function
        histgridRef = np.zeros((gray.shape[0]//blockH+1, gray.shape[1]//blockW+1, 256), dtype=np.float32)
        for i in range(0, gray.shape[0]):
            for j in range(0, gray.shape[1]):
                # soft histogramming
                b00idx = j // blockW
                b00x = b00idx * blockW
                b00idy = i // blockH
                b00y = b00idy * blockH

                s = (j - b00x) / blockW
                t = (i - b00y) / blockH

                v = gray[i, j]

                histgridRef[b00idy, b00idx, v] += sqrt((1-s)**2 + (1-t)**2)
                histgridRef[b00idy, b00idx+1, v] += sqrt(s**2 + (1-t)**2)
                histgridRef[b00idy+1, b00idx, v] += sqrt((1-s)**2 + t**2)
                histgridRef[b00idy+1, b00idx+1, v] += sqrt(s**2 + t**2)
        
        print('diff {}'.format(histgridRef.size - np.sum(np.isclose(histgrid, histgridRef))))

        # apply transfer function
    else:
        for i in range(0, gray.shape[0]):
            for j in range(0, gray.shape[1]):
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
                mappings[i,j,:] = calc_transfer_func(histgrid[i, j], alpha, punch, clip).astype(np.float32)

        for i in range(0, gray.shape[0]):
            for j in range(0, gray.shape[1]):
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

    return gray
