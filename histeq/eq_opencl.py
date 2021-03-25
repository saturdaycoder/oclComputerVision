import pyopencl as cl
import cv2
import numpy as np
import os

def get_elapsed_ms(e):
    return (e.profile.end - e.profile.start) / 1000000
class clHistEq:
    __inst = None

    def __init__(self):
        self.histBins = 256
        self.histThreads = 32
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        #os.environ['PYOPENCL_NO_CACHE'] = '1'
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
                        self.clKernHistEqLocalBlock = self.clPrg.histeq_local_block

    @classmethod
    def getInstance(cls):
        if cls.__inst == None:
            cls.__inst = clHistEq()
        return cls.__inst

    def histGrid(self, gray):
        mf = cl.mem_flags
        w = gray.shape[1]
        h = gray.shape[0]
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        clImgIn = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        groupw = w // self.histBins
        grouph = h // self.histThreads
        npHistOut = np.zeros((grouph, groupw, self.histBins), dtype=np.uint32)
        clHistOut = cl.Buffer(self.clCtx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=npHistOut)
        self.clKernHist.set_args(clImgIn, clHistOut)
        ev0 = cl.enqueue_copy(self.clQueue, clImgIn, gray, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernHist, (w//self.histBins, h), (1, self.histThreads), wait_for=[ev0])
        ev1.wait()
        return npHistOut, get_elapsed_ms(ev0) + get_elapsed_ms(ev1)

    def histeqGlobal(self, gray, mapping):
        mf = cl.mem_flags
        w = gray.shape[1]
        h = gray.shape[0]
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        clImgIn = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        npImgOut = np.empty_like(gray)
        clImgOut = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(w, h))
        clMapping = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mapping)

        self.clKernHistEqGlobal.set_args(clImgIn, clImgOut, clMapping)
        ev0 = cl.enqueue_copy(self.clQueue, clImgIn, gray, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernHistEqGlobal, (w, h), (16, 16), wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, npImgOut, clImgOut, origin=(0, 0), region=(w, h), wait_for=[ev1])
        ev2.wait()
        return npImgOut, get_elapsed_ms(ev0) + get_elapsed_ms(ev1) + get_elapsed_ms(ev2)

    def histeqLocalBlock(self, gray, mappings, blockshape):
        mf = cl.mem_flags
        w = gray.shape[1]
        h = gray.shape[0]
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        clImgIn = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        npImgOut = np.empty_like(gray)
        clImgOut = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(w, h))
        clMappings = cl.Buffer(self.clCtx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mappings.astype(np.float32))
        clBlockWidth = np.int32(blockshape[1])
        clBlockHeight = np.int32(blockshape[0])
        clBlockNumX = np.int32(mappings.shape[1])
        clBlockNumY = np.int32(mappings.shape[0])

        self.clKernHistEqLocalBlock.set_args(clImgIn, clImgOut, clMappings, clBlockWidth, clBlockHeight, clBlockNumX, clBlockNumY)
        ev0 = cl.enqueue_copy(self.clQueue, clImgIn, gray, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernHistEqLocalBlock, (w, h), (16, 16), wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, npImgOut, clImgOut, origin=(0, 0), region=(w, h), wait_for=[ev1])
        ev2.wait()
        return npImgOut, get_elapsed_ms(ev0) + get_elapsed_ms(ev1) + get_elapsed_ms(ev2)