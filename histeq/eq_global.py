import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import pyopencl as cl
import os
import matplotlib.pyplot as plt

def histeq_global(bgr, alpha=1, punch=0.05, clip=2):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    gray = ycrcb[:,:,0].copy()
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
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
    #I = (X - dark_punch) / (bright_punch - dark_punch) * 255
    #I[:dark_punch] = 0
    #I[bright_punch:] = 255
    mapping = alpha * cdf * 255 + (1-alpha) * I
    # clipping
    mapping = np.clip(mapping, 0, 255)
    #plt.title('Transfer function')
    #plt.plot(X, cdf * 255, color='green', label='CDF')
    #plt.plot(X, I, color='red', label='I')
    #plt.plot(X, mapping, color='blue', label='mapping')
    #plt.legend()
    #plt.xlabel('Original')
    #plt.ylabel('Transferred')
    #plt.show()

    # gain limiting with clip
    mapping = np.clip(mapping, I / clip, I * clip).astype(np.uint8)

    eq = lambda t: mapping[t]
    vfunc = np.vectorize(eq)
    gray = vfunc(gray)

    ycrcb[:,:,0] = gray
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
