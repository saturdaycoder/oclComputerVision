import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
import time
import pyopencl as cl
import os

class clHistogram:
    def __init__(self, bins = 256):
        self.bins = 256
        self.threadNum = 32
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
        for p in cl.get_platforms():
            if 'AMD' in p.name:
                devices = p.get_devices(device_type=cl.device_type.GPU)
                if len(devices) > 0:
                    self.clDev = devices[0]
                    self.clCtx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, p)])
                    self.clQueue = cl.CommandQueue(self.clCtx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                    with open("hist.cl","r") as f:
                        kernSrc = f.read()
                        self.clPrg = cl.Program(self.clCtx, kernSrc).build(options = ' -DBINS={} -DTHREAD_NUM={} -DN={} '.format(self.bins, self.threadNum, self.bins//self.threadNum))
                        self.clKern = self.clPrg.hist

    def hist(self, im):
        mf = cl.mem_flags
        clImgIn = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=im)
        #clImgIn = cl.Buffer(self.clCtx, mf.READ_ONLY, im.nbytes)
        width = im.shape[1]
        height = im.shape[0]
        groupw = width // self.bins
        grouph = height // self.threadNum
        npHistOut = np.zeros((grouph, groupw, self.bins), dtype=np.uint32)
        clHistOut = cl.Buffer(self.clCtx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=npHistOut)
        #clHistOut = cl.Buffer(self.clCtx, mf.READ_WRITE, npHistOut.nbytes)
        clWidth = np.int32(width)
        clHeight = np.int32(height)
        t1 = time.time()
        self.clKern.set_args(clImgIn, clWidth, clHeight, clHistOut)
        #evt = cl.enqueue_copy(self.clQueue, clImgIn, im)
        evt = cl.enqueue_nd_range_kernel(self.clQueue, self.clKern, (height, width//self.bins), (self.threadNum, 1))
        #evt = cl.enqueue_copy(self.clQueue, npHistOut, clHistOut)
        evt.wait()
        return npHistOut.sum(axis=0).sum(axis=0).astype(np.int64), evt.profile.end - evt.profile.start

if __name__ == '__main__':
    clHist = clHistogram(bins=256)
    im = cv2.imread('lenna.png')
    im = cv2.resize(im, (1024, 1024))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    loop = 20

    t1 = time.time()
    for j in range(0, loop):
        histRef, _ = np.histogram(gray, bins=256, range=(0, 256))

    t2 = time.time()
    prof_elapsed = 0
    for j in range(0, loop):
        histCl, elapsed = clHist.hist(gray)
        prof_elapsed += elapsed
    t3 = time.time()

    #assert(np.sum(histRef) == im.shape[0] * im.shape[1])
    #assert(np.sum(histCl) == im.shape[0] * im.shape[1])

    print('CPU: {:.3f} ms, GPU: {:.3f} ms/{:.3f} ms'.format((t2-t1)*1000/loop, (t3-t2)*1000/loop, prof_elapsed/1000000/loop))

    print('same {}'.format(np.sum(histRef == histCl)))

    imFig, imAx = plt.subplots(2, 1, figsize=(19, 12))
    imAx[0].bar(np.arange(len(histRef)), histRef)
    imAx[0].set_title('numpy')
    imAx[1].bar(np.arange(len(histCl)), histCl)
    imAx[1].set_title('opencl')
    plt.show()
