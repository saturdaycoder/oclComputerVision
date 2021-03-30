import pyopencl as cl
import numpy as np
import cv2
import os
import sys
import time
import pickle
import numpy as np
from math import atan2, floor, pi
from skimage.metrics import peak_signal_noise_ratio

def get_elapsed_ms(ev_list):
    elapsed = []
    for e in ev_list:
        elapsed.append((e.profile.end - e.profile.start) / 1000000)
    return elapsed

class ClRaisr:
    workGroupSize = (16, 16)
    cscRgba2yuvaBT601 = np.array([
        0.299, 0.587, 0.114, 0.0,
        -0.14713, -0.28886, 0.436, 0.5,
        0.615, -0.51499, 0.10001, 0.5,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    cscYuva2rgbaBT601 = np.array([
        1.0, 0.0, 1.13983, -1.13983 * 0.5,
        1.0, -0.39465, -0.58060, (0.39465 + 0.58060) * 0.5,
        1.0, 2.03211, 0.0, -2.03211 * 0.5,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    cscRgba2yuvaBT709 = np.array([
        0.2126, 0.7152, 0.0722, 0.0,
        -0.09991, -0.33609, 0.436, 0.5,
        0.615, -0.55861, 0.05639, 0.5,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    cscYuva2rgbaBT709 = np.array([
        1.0, 0.0, 1.28033, -1.28033 * 0.5,
        1.0, -0.21482, -0.38059, (0.21482 + 0.38059) * 0.5,
        1.0, 2.12798, 0.0, -2.12798 * 0.5,
        0.0, 0.0, 0.0, 1.0
    ], dtype=np.float32)
    sobelX = np.array([
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ], dtype=np.float32)
    sobelY = np.array([
        -1.0, -2.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 1.0
    ], dtype=np.float32)
    def gaussian2d(self, shape=(3,3),sigma=0.5):
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

    def __init__(self, grayMode):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
        self.grayMode = grayMode
        for p in cl.get_platforms():
            if 'AMD' in p.name:
                devices = p.get_devices(device_type=cl.device_type.GPU)
                if len(devices) > 0:
                    self.dev = devices[0]
                    self.ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, p)])
                    self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                    filepath = os.path.split(os.path.realpath(__file__))[0]
                    with open(os.path.join(filepath, 'raisr.cl'), 'r') as f:
                        kernSrc = f.read()
                        self.prg = cl.Program(self.ctx, kernSrc).build(options = ' -DGRAY_MODE={} -DWORK_GROUP_WIDTH={} -DWORK_GROUP_HEIGHT={} '.format(self.grayMode, self.workGroupSize[1], self.workGroupSize[0]))
        with open(os.path.join(filepath, 'filter.p'), 'rb') as fp:
            self.filters_x2 = pickle.load(fp).astype(np.float32)
        
        self.gaussian = self.gaussian2d([9, 9], 2)
        self.gaussian = np.diag(self.gaussian.ravel()).astype(np.float32)
        self.gaussian = np.diag(self.gaussian).copy()
        self.kern = self.prg.raisr

    def upsample(self, src, dst, scale_factor):
        srcw = src.shape[1]
        srch = src.shape[0]
        dstw = dst.shape[1]
        dsth = dst.shape[0]
        if scale_factor == 2:
            filters = self.filters_x2
        else:
            print('Fatal. not trained for scale factor {}'.format(scale_factor))
            return

        mf = cl.mem_flags
        if self.grayMode == 1:
            fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNORM_INT8)
        else:
            fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)
        clSrcImg = cl.Image(self.ctx, mf.READ_ONLY, fmt, shape=(srcw, srch))
        clDstImg = cl.Image(self.ctx, mf.WRITE_ONLY, fmt, shape=(dstw, dsth))

        clSobelX = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.sobelX)
        clSobelY = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.sobelY)
        clCscRgba2yuva = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.cscRgba2yuvaBT601)
        clCscYuva2rgba = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.cscYuva2rgbaBT601)
        clGaussian = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=self.gaussian)
        clFilters = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=filters)
        npStreQuantizer = np.array([0.0001, 0.001], dtype=np.float32)
        clStreQ = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=npStreQuantizer)
        npCoheQuantizer = np.array([0.25, 0.5], dtype=np.float32)
        clCoheQ = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=npCoheQuantizer)

        self.kern.set_args( \
            clSrcImg, \
            clDstImg, \
            clSobelX, \
            clSobelY, \
            clCscRgba2yuva, \
            clCscYuva2rgba, \
            clGaussian, \
            clStreQ, \
            clCoheQ, \
            np.int32(scale_factor), \
            clFilters \
        )
        ev0 = cl.enqueue_copy(self.queue, clSrcImg, src, origin=(0, 0), region=(srcw, srch))
        ev1 = cl.enqueue_nd_range_kernel(self.queue, self.kern, (dstw, dsth), (16, 16), wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.queue, dst, clDstImg, origin=(0, 0), region=(dstw, dsth), wait_for=[ev1])
        ev2.wait()

        return get_elapsed_ms([ev0, ev1, ev2])

if __name__ == '__main__':

    imgGray = 0

    raisr = ClRaisr(imgGray)

    bgr = cv2.imread('images/Set5/img_001_SRF_2_LR.png')
    #bgr = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    w = bgr.shape[1]
    h = bgr.shape[0]
    wnew = 2 * w
    hnew = 2 * h
    refHR = cv2.imread('images/Set5/img_001_SRF_2_HR.png')
    refUp = cv2.resize(bgr, (wnew, hnew), interpolation=cv2.INTER_CUBIC)

    loopcount = 20
    count = 0

    if imgGray == 1:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb_dst = cv2.resize(ycrcb, (wnew, hnew), interpolation=cv2.INTER_LINEAR)
        src = ycrcb[:,:,0].copy()
        dst = np.zeros((hnew, wnew), dtype=src.dtype)
    else:
        src = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        dst = np.zeros((hnew, wnew, 4), dtype=np.uint8)

    elapsed_list = None
    while count < loopcount:
        elapsed = raisr.upsample(src, dst, 2)
        if elapsed_list is None:
            elapsed_list = elapsed
        else:
            for i in range(len(elapsed)):
                elapsed_list[i] += elapsed[i]
        count += 1

    if imgGray == 1:
        ycrcb_dst[:,:,0] = dst
        dst = cv2.cvtColor(ycrcb_dst, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite('raisr-out.png', dst)
    cv2.imwrite('raisr-ref-upsample.png', refUp)
    print('elapsed: {:.3f} + {:.3f} + {:.3f} ms'.format(elapsed_list[0]/count, elapsed_list[1]/count, elapsed_list[2]/count))

    psnrCubic = peak_signal_noise_ratio(refUp, refHR, data_range=255)
    psnrRaisr = peak_signal_noise_ratio(dst[:,:,0:3], refHR, data_range=255)
    print('PSNR: cubic {:.3f} raisr {:.3f}'.format(psnrCubic, psnrRaisr))
