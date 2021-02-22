from histeq.eq_global import histeq_global
from histeq.eq_local_block import histeq_local_block
from histeq.eq_local_corner import histeq_local_corner
import cv2
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from eq_opencl import clHistEq

alphaMax = 100
alphaPos = 30
punchMax = 100
punchPos = 5
clip = 2
clipMax = 10

algoList = [
    'Global Histogram Eqaulization',
    'Local Histogram Equalization (block-based)',
    #'Local Histogram Equalization (corner-based)',
    'OpenCV CLAHE'
]
algoId = 1

infoList = [
    'Nothing',
    'Algo',
    'Algo + Hist',
    #'Algo + Hist + FPS'
]
infoId = 2

def onAlphaChanged(x):
    global alphaPos
    alphaPos = x

def onPunchChanged(x):
    global punchPos
    punchPos = x

def onClipChanged(x):
    global clip
    clip = x

def onAlgoChanged(x):
    global algoId
    algoId = x

def onInfoChanged(x):
    global infoId
    infoId = x

windowName = 'HistEq Demo, press ESC to quit'
cv2.namedWindow(windowName)
cv2.createTrackbar('alpha(%)', windowName, alphaPos, alphaMax, onAlphaChanged)
cv2.createTrackbar('punch(%)', windowName, punchPos, punchMax, onPunchChanged)
cv2.createTrackbar('clipp', windowName, clip, clipMax, onClipChanged)
cv2.createTrackbar('algo', windowName, algoId, len(algoList)-1, onAlgoChanged)
cv2.createTrackbar('info', windowName, infoId, len(infoList)-1, onInfoChanged)

cap = cv2.VideoCapture('video/HDR_outdoor_720P.mp4')
clahe = cv2.createCLAHE(tileGridSize=(3, 5), clipLimit=2)
cleq = clHistEq.getInstance()
def plotHist(img):
    B = img[:,:,0].copy()
    G = img[:,:,1].copy()
    R = img[:,:,2].copy()
    histGrid, evt = cleq.histGrid(B)
    evt.wait()
    hist_B = histGrid.sum(axis=0).sum(axis=0)
    histGrid, evt = cleq.histGrid(G)
    evt.wait()
    hist_G = histGrid.sum(axis=0).sum(axis=0)
    histGrid, evt = cleq.histGrid(R)
    evt.wait()
    hist_R = histGrid.sum(axis=0).sum(axis=0)
    hist_height = 100
    hist_width = 256
    hist_left = 10
    hist_B_top = img.shape[0] - 10 - hist_height
    hist_G_top = hist_B_top - 10 - hist_height
    hist_R_top = hist_G_top - 10 - hist_height
    bin_width = hist_width/len(hist_B)
    hist_max = np.max((hist_B, hist_G, hist_R))
    for x,y in enumerate(hist_B):
        left = (int)(hist_left+x*bin_width)
        right = (int)(hist_left+(x+1)*bin_width)
        top = (int)(hist_B_top+(1-y/hist_max)*hist_height)
        bottom = (int)(hist_B_top+hist_height)
        img = cv2.rectangle(img, (left, top), (right, bottom), (255,0,0))
    for x,y in enumerate(hist_G):
        left = (int)(hist_left+x*bin_width)
        right = (int)(hist_left+(x+1)*bin_width)
        top = (int)(hist_G_top+(1-y/hist_max)*hist_height)
        bottom = (int)(hist_G_top+hist_height)
        img = cv2.rectangle(img, (left, top), (right, bottom), (0,255,0))
    for x,y in enumerate(hist_R):
        left = (int)(hist_left+x*bin_width)
        right = (int)(hist_left+(x+1)*bin_width)
        top = (int)(hist_R_top+(1-y/hist_max)*hist_height)
        bottom = (int)(hist_R_top+hist_height)
        img = cv2.rectangle(img, (left, top), (right, bottom), (0,0,255))
    return img

def addText(img, text):
    overlay_image = np.copy(img)
    out_img = img.copy()
    cv2.rectangle(img=overlay_image, pt1=(5, 5), pt2=(500, 50),
        color=[0, 0, 0], thickness=-1)
    cv2.putText(img=overlay_image, text=text, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7, color=(255,255,255), thickness=2)
    return cv2.addWeighted(src1=overlay_image, alpha=0.7, src2=img, beta=0.3, gamma=0, dst=out_img)

while True:
    ret, im = cap.read()
    if not ret:
        print('end of video, reset to beginning')
        frame_id = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, im = cap.read()
    #im = cv2.imread('images/under_exposure.jpg')
    im = cv2.resize(im, (1280, 768))

    ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    gray = ycrcb[:,:,0].copy()

    if algoId == 0:
        gray_new = histeq_global(gray, alpha=alphaPos/alphaMax, punch=punchPos/punchMax, clip=clip)
    elif algoId == 1:
        gray_new = histeq_local_block(gray, alpha=alphaPos/alphaMax, punch=punchPos/punchMax, clip=clip)
    #elif algoId == 2:
    #    gray_new = histeq_local_corner(gray, alpha=alphaPos/alphaMax, punch=punchPos/punchMax, clip=clip)
    else:
        gray_new = clahe.apply(gray)
    ycrcb[:,:,0] = gray_new
    im_new = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if infoId >= 1:
        im_new = addText(im_new, algoList[algoId])

    if infoId >= 2:
        im = plotHist(im)
        im_new = plotHist(im_new)

    disp = np.concatenate((im, im_new), axis=1)

    cv2.imshow(windowName, disp)
    key = cv2.waitKey(1)
    if key == 27:
        print('exit')
        break