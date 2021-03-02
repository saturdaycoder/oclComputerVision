import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import os

def gaussian_pyramid(img, scale=2, depth=3):
    pyramid = []
    pyramid.insert(0, img)
    src = img
    for i in range (0, depth-1):
        src = cv2.pyrDown(src, dstsize=(src.shape[1] // scale, src.shape[0] // scale))
        pyramid.insert(0, src)
    return pyramid

if __name__ == '__main__':
    img = cv2.imread('images/lenna.png')
    img = cv2.resize(img, (1280, 720))
    t1 = time.time()
    pyr = gaussian_pyramid(img, 2, 3)
    print('spent {:.3f} ms'.format((time.time() - t1)*1000))

    i = 0
    for im in pyr:
        cv2.imshow('layer {}'.format(i), im)
        i += 1
    cv2.waitKey()
