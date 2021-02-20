import pyopencl as cl
import cv2
import numpy as np
import os

class clHistEq:
    __inst = None

    def __init__(self):
        self.histBins = 256
        self.histThreads = 32
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
        for p in cl.get_platforms():
            if 'AMD' in p.name:
                devices = p.get_devices(device_type=cl.device_type.GPU)
                if len(devices) > 0:
                    self.clDev = devices[0]
                    self.clCtx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, p)])
                    self.clQueue = cl.CommandQueue(self.clCtx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                    filepath = os.path.split(os.path.realpath(__file__))[0]
                    with open(os.path.join(filepath, 'hist.cl'), 'r') as f:
                        kernSrc = f.read()
                        self.clPrg = cl.Program(self.clCtx, kernSrc).build(options = ' -DHIST_BINS={} -DHIST_THREAD_NUM={} -DHIST_N={} '.format(self.histBins, self.histThreads, self.histBins // self.histThreads))
                        self.clKernHist = self.clPrg.hist
                        self.clKernHistEqGlobal = self.clPrg.histeq_global

    @classmethod
    def getInstance(cls):
        if cls.__inst == None:
            cls.__inst = clHistEq()
        return cls.__inst

    def histGrid(self, gray, wait_for=None):
        mf = cl.mem_flags
        clImgIn = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=gray)
        width = gray.shape[1]
        height = gray.shape[0]
        groupw = width // self.histBins
        grouph = height // self.histThreads
        npHistOut = np.zeros((grouph, groupw, self.histBins), dtype=np.uint32)
        clHistOut = cl.Buffer(self.clCtx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=npHistOut)
        clWidth = np.int32(width)
        clHeight = np.int32(height)
        self.clKernHist.set_args(clImgIn, clWidth, clHeight, clHistOut)
        evt = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernHist, (height, width//self.histBins), (self.histThreads, 1), wait_for=wait_for)
        return npHistOut, evt

    def histeqGlobal(self, gray, mapping, wait_for=None):
        mf = cl.mem_flags
        clImgIn = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=gray)
        npImgOut = gray.copy()
        clImgOut = cl.Buffer(self.clCtx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=npImgOut)
        width = gray.shape[1]
        height = gray.shape[0]
        clMapping = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mapping)
        clWidth = np.int32(width)
        clHeight = np.int32(height)
        self.clKernHistEqGlobal.set_args(clImgIn, clWidth, clHeight, clImgOut, clMapping)
        evt = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernHistEqGlobal, (height, width), (16, 16), wait_for=wait_for)
        return npImgOut, evt