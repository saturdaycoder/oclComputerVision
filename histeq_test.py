from histeq.eq_global import histeq_global
from histeq.eq_local_block import histeq_local_block
#from histeq.eq_local_corner import histeq_local_corner
import cv2
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5
punch = 0.05
clip = 3

alphaMax = 100
alphaPos = 50
punchMax = 100
punchPos = 5
clipMax = 10

def onAlphaChanged(x):
    global alpha
    global alphaPos
    global alphaMax
    alphaPos = x
    alpha = alphaPos / alphaMax

def onPunchChanged(x):
    global punch
    global punchPos
    global punchMax
    punchPos = x
    punch = punchPos / punchMax

def onClipChanged(x):
    global clip
    clip = x

cv2.namedWindow('histeq')
cv2.createTrackbar('alpha', 'histeq', alphaPos, alphaMax, onAlphaChanged)
cv2.createTrackbar('punch', 'histeq', punchPos, punchMax, onPunchChanged)
cv2.createTrackbar('clip', 'histeq', clip, clipMax, onClipChanged)

cap = cv2.VideoCapture('video/HDR_outdoor_720P.mp4')
ret, im = cap.read()
#im = cv2.imread('images/under_exposure.jpg')
im = cv2.resize(im, (1152, 768))

while True:
    im_new = histeq_global(im, alpha=alpha, punch=punch, clip=clip)

    '''
    imFig, imAx = plt.subplots(2, 2, figsize=(25, 12))
    imFig.tight_layout()
    imAx[0, 0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    imAx[0, 1].imshow(cv2.cvtColor(im_new, cv2.COLOR_BGR2RGB))
    hist, _ = np.histogram(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), bins=256, range=(0,256))
    hist_new, _ = np.histogram(cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY), bins=256, range=(0,256))
    imAx[1, 0].bar(np.arange(len(hist)), hist)
    imAx[1, 1].bar(np.arange(len(hist_new)), hist_new)
    plt.show()
    '''

    disp = np.concatenate((im, im_new), axis=1)

    cv2.imshow('histeq', disp)
    cv2.waitKey(1)