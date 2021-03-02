import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import os, sys

def mv2hsv(mv, scale=1):
    mag, ang = cv2.cartToPolar(mv[..., 0], mv[..., 1])
    hsv = np.zeros((mv.shape[0], mv.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (hsv.shape[1]*scale, hsv.shape[0]*scale))

def gaussian2d(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def SSD(p0, p1):
    assert len(p0.shape) == 2
    assert p0.shape == p1.shape
    diff = p0.astype(np.float32) - p1.astype(np.float32)
    diffsquare = np.power(diff, 2)
    return (float)(np.sum(diffsquare))

def SAD(p0, p1):
    assert len(p0.shape) == 2
    assert p0.shape == p1.shape
    diff = p0.astype(np.float32) - p1.astype(np.float32)
    diffabsolute = np.absolute(diff)
    return (float)(np.sum(diffabsolute))

def WSAD(p0, p1, sigma=2.0):
    weights = gaussian2d(p0.shape, sigma)
    assert len(p0.shape) == 2
    assert p0.shape == p1.shape
    diffabsolute = np.absolute(np.dot(p0.astype(np.float32), weights) - np.dot(p1.astype(np.float32), weights))
    return (float)(np.sum(diffabsolute))

visualizeSearch = False
def get_displacement(patch, searchRegion):
    if searchRegion.shape[0] < patch.shape[0] or searchRegion.shape[1] < patch.shape[1]:
        return 0, 0
    
    minSAD = sys.float_info.max
    minPos = (-1, -1)
    searchCenter = (searchRegion.shape[0]//2 - patch.shape[0]//2, searchRegion.shape[1]//2 - patch.shape[1]//2)

    # visualize >>>>>>>
    if visualizeSearch:
        scale = 30
        bgrPatch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        bgrPatch = cv2.resize(bgrPatch, (bgrPatch.shape[1]*scale, bgrPatch.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('patch', bgrPatch)
        cv2.waitKey(1)
    # visualize <<<<<<<

    for row in range(0, searchRegion.shape[0], patch.shape[1]):
        for col in range(0, searchRegion.shape[1], patch.shape[1]):
            sad = SAD(patch, searchRegion[row:row+patch.shape[0], col:col+patch.shape[1]])
            if sad < minSAD:
                minSAD = sad
                minPos = (row, col)
            
            # visualize >>>>>>>
            if visualizeSearch:
                bgrSearch = cv2.cvtColor(searchRegion, cv2.COLOR_GRAY2BGR)
                bgrSearch = cv2.resize(bgrSearch, (bgrSearch.shape[1]*scale, bgrSearch.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
                bgrMatch = cv2.rectangle(bgrSearch, (col*scale, row*scale), ((col+patch.shape[1])*scale, (row+patch.shape[0])*scale), (0, 0, 255), thickness=1)
                bgrMatch = cv2.rectangle(bgrMatch, (minPos[1]*scale, minPos[0]*scale), ((minPos[1]+patch.shape[1])*scale, (minPos[0]+patch.shape[0])*scale), (0, 255, 0), thickness=1)
                print('pos = ({}, {}), SAD={:.3f}, minSAD={:.3f}'.format(col-searchCenter[1], row-searchCenter[0], sad, minSAD))
                cv2.imshow('search', bgrMatch)
                cv2.waitKey()
            # visualize <<<<<<<
    
    assert (minPos[0] >= 0 and minPos[1] >= 0)
    return minPos[0] - searchCenter[0], minPos[1] - searchCenter[1]

def get_region(im, T, B, L, R, size):
    marginT = 0
    marginB = 0
    marginL = 0
    marginR = 0
    if B < 0:
        T = B = 0
        marginT = size
        marginB = 0
    elif T < 0 and B >= 0:
        T = 0
        marginT = size - B
        marginB = 0
    elif T <= im.shape[0] and B > im.shape[0]:
        B = im.shape[0]
        marginT = 0
        marginB = size - (im.shape[0] - T)
    elif T > im.shape[0]:
        T = B = im.shape[0]
        marginT = 0
        marginB = size
    if R < 0:
        L = R = 0
        marginL = size
        marginR = 0
    elif L < 0 and R >= 0:
        L = 0
        marginL = size - R
        marginR = 0
    elif L <= im.shape[1] and R > im.shape[1]:
        R = im.shape[1]
        marginL = 0
        marginR = size - (im.shape[1] - L)
    elif L > im.shape[1]:
        L = R = im.shape[1]
        marginL = 0
        marginR = size
    region = im[T:B, L:R]
    return cv2.copyMakeBorder(region, marginT, marginB, marginL, marginR, cv2.BORDER_CONSTANT, value=0)

visualizeME = False
def estimate_motion_vector(gray0, gray1, searchSize=15, patchSize=5, seed=None, pyrScale=1):
    searchMargin = searchSize // 2
    patchMargin = patchSize // 2
    searchBlockSize = searchSize

    if seed is None:
        mv = np.zeros((gray0.shape[0], gray0.shape[1], 2), dtype=np.float32)
    else:
        mv = seed.copy()

    for row in range(0, gray0.shape[0]-patchSize):
        for col in range(0, gray0.shape[1]-patchSize):
            patch = gray0[row:row+patchSize, col:col+patchSize]
            T = row + int(mv[row, col, 0]) - searchMargin + patchMargin
            L = col + int(mv[row, col, 1]) - searchMargin + patchMargin
            R = L + searchBlockSize
            B = T + searchBlockSize

            searchRegion = get_region(gray1, T, B, L, R, searchBlockSize)
            d = get_displacement(patch, searchRegion)

            # visualize ME >>>>>>>
            if visualizeME:
                scale = 3
                # search bounding box
                bgr0 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
                bgr1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
                bgr0 = cv2.resize(bgr0, (bgr0.shape[1]*scale, bgr0.shape[0]*scale))
                bgr1 = cv2.resize(bgr1, (bgr1.shape[1]*scale, bgr1.shape[0]*scale))
                im0 = cv2.rectangle(bgr0, (L*scale, T*scale), (R*scale, B*scale), (0, 255, 255), thickness=1)
                im1 = cv2.rectangle(bgr1, (L*scale, T*scale), (R*scale, B*scale), (0, 255, 255), thickness=1)
                # patch bounding box
                im0 = cv2.rectangle(im0, (col*scale, row*scale), ((col+patchSize)*scale, (row+patchSize)*scale), (0, 0, 255), thickness=1)
                im1 = cv2.rectangle(im1, (col*scale, row*scale), ((col+patchSize)*scale, (row+patchSize)*scale), (0, 0, 255), thickness=1)
                # matched bounding box
                im0 = cv2.rectangle(im0, ((col+d[1])*scale, (row+d[0])*scale), ((col+patchSize+d[1])*scale, (row+patchSize+d[0])*scale), (0, 255, 0), thickness=1)
                im1 = cv2.rectangle(im1, ((col+d[1])*scale, (row+d[0])*scale), ((col+patchSize+d[1])*scale, (row+patchSize+d[0])*scale), (0, 255, 0), thickness=1)
                # zoomed in
                block0 = gray0[row:row+patchSize, col:col+patchSize]
                block1 = gray1[row+d[1]:row+d[1]+patchSize, col+d[0]:col+d[0]+patchSize]
                block0 = cv2.cvtColor(block0, cv2.COLOR_GRAY2BGR)
                block1 = cv2.cvtColor(block1, cv2.COLOR_GRAY2BGR)
                block0 = cv2.resize(block0, (im0.shape[1], im0.shape[1]), interpolation=cv2.INTER_NEAREST)
                block1 = cv2.resize(block1, (im0.shape[1], im0.shape[1]), interpolation=cv2.INTER_NEAREST)
                # display
                bar = np.ones((im0.shape[0], 2, 3), dtype=np.uint8) * 255
                row0 = np.concatenate((im0, bar, im1), axis=1)
                bar = np.ones((block0.shape[0], 2, 3), dtype=np.uint8) * 255
                row1 = np.concatenate((block0, bar, block1), axis=1)
                bar = np.ones((2, row0.shape[1], 3), dtype=np.uint8) * 255
                disp = np.concatenate((row0, bar, row1), axis=0)
                cv2.imshow("ME", disp)
                cv2.waitKey()
            # visualize ME <<<<<<<

            mv[row, col, 0] += d[1]
            mv[row, col, 1] += d[0]

            if visualizeME:
                bgr = mv2hsv(mv, scale=pyrScale)
                cv2.imshow('estimate', bgr)
                cv2.waitKey(1)

    return mv

