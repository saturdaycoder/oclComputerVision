import numpy as np
import cv2
from math import ceil, floor, sqrt
import time
import os
import matplotlib.pyplot as plt
from pyramid.pyramid import gaussian_pyramid
from motion_estimation.me_pyramid import estimate_motion_vector, mv2hsv
import os

TAG_FLOAT = 202021.25
def read_flo(file):
	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = (int)(np.fromfile(f, np.int32, count=1))
	h = (int)(np.fromfile(f, np.int32, count=1))
	data = np.fromfile(f, np.float32, count=2*w*h)
	flow = np.resize(data, (int(h), int(w), 2))	
	f.close()
	return flow

TAG_STRING = b'PIEH'
def write_flo(flow, filename):
	assert type(filename) is str, "file is not str %r" % str(filename)
	assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	height, width, nBands = flow.shape
	assert nBands == 2, "Number of bands = %r != 2" % nBands
	u = flow[: , : , 0]
	v = flow[: , : , 1]	
	assert u.shape == v.shape, "Invalid flow shape"
	height, width = u.shape
	f = open(filename,'wb')
	f.write(TAG_STRING)
	np.array(width).astype(np.int32).tofile(f)
	np.array(height).astype(np.int32).tofile(f)
	tmp = np.zeros((height, width*nBands))
	tmp[:,np.arange(width)*2] = u
	tmp[:,np.arange(width)*2 + 1] = v
	tmp.astype(np.float32).tofile(f)
	f.close()

def show_mv(name, mv, scale=1):
    bgr = mv2hsv(mv, scale)
    cv2.imshow(name, bgr)
    cv2.waitKey(1)

def upscale_mv(mv, scale):
    width = mv.shape[1]
    height = mv.shape[0]
    mv = mv.transpose(2, 0, 1)
    u = mv[0]
    v = mv[1]
    uMax = np.max(u)
    vMax = np.max(v)
    u = cv2.resize(u / uMax, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v / vMax, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    u *= uMax * scale
    v *= vMax * scale
    return np.stack((u, v), axis=-1)

im0 = cv2.imread('images/frame10.png')
im1 = cv2.imread('images/frame11.png')
gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
mv_gt = read_flo('images/flow10.flo')
write_flo(mv_gt, 'test.flo')
show_mv('Grond-truth', mv_gt)

pyr0 = gaussian_pyramid(gray0, 2, 3)
pyr1 = gaussian_pyramid(gray1, 2, 3)

if True:
	mv0 = estimate_motion_vector(pyr0[0], pyr1[0], 15, 5, seed=None, pyrScale=4)
	write_flo(mv0, 'layer0.flo')
else:
	mv0 = read_flo('layer0.flo')
show_mv('Layer 0', mv0, scale=4)

mv1 = estimate_motion_vector(pyr0[1], pyr1[1], 15, 5, seed=upscale_mv(mv0, 2), pyrScale=2)
show_mv('Layer 1', mv1, scale=2)
write_flo(mv1, 'layer1.flo')

mv2 = estimate_motion_vector(pyr0[2], pyr1[2], 15, 5, seed=upscale_mv(mv1, 2), pyrScale=1)
show_mv('Layer 2', mv2, scale=1)
write_flo(mv2, 'layer2.flo')

cv2.waitKey()