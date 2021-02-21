from histeq.eq_global import histeq_global
from histeq.eq_local_block import histeq_local_block
#from histeq.eq_local_corner import histeq_local_corner
import cv2
import numpy as np
import matplotlib.pyplot as plt

alphaMax = 100
alphaPos = 30
punchMax = 100
punchPos = 5
clip = 2
clipMax = 10

def onAlphaChanged(x):
    global alphaPos
    alphaPos = x

def onPunchChanged(x):
    global punchPos
    punchPos = x

def onClipChanged(x):
    global clip
    clip = x

cv2.namedWindow('histeq')
cv2.createTrackbar('alpha(%)', 'histeq', alphaPos, alphaMax, onAlphaChanged)
cv2.createTrackbar('punch(%)', 'histeq', punchPos, punchMax, onPunchChanged)
cv2.createTrackbar('clip', 'histeq', clip, clipMax, onClipChanged)

cap = cv2.VideoCapture('video/HDR_outdoor_720P.mp4')

while True:
    ret, im = cap.read()
    if not ret:
        print('end of video, reset to beginning')
        frame_id = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, im = cap.read()
    #im = cv2.imread('images/under_exposure.jpg')
    im = cv2.resize(im, (1280, 768))

    im_new = histeq_local_block(im, alpha=alphaPos/alphaMax, punch=punchPos/punchMax, clip=clip)

    disp = np.concatenate((im, im_new), axis=1)

    cv2.imshow('histeq', disp)
    cv2.waitKey(1)