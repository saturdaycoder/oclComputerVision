import numpy as np
import cv2
import pyopencl as cl
import os, sys
from scipy.interpolate import interp2d
from skimage.metrics import peak_signal_noise_ratio
import time


def get_elapsed_ms(ev_list):
    elapsed = []
    for e in ev_list:
        elapsed.append((e.profile.end - e.profile.start) / 1000000)
    return elapsed

class clUtility:
    def __init__(self):
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
                    with open(os.path.join(filepath, 'interpolation.cl'), 'r') as f:
                        kernSrc = f.read()
                        self.clPrg = cl.Program(self.clCtx, kernSrc).build(options = ' -DMAX_LOCAL_WIDTH={} -DMAX_LOCAL_HEIGHT={}  '.format(18, 18))
                        self.clKernBilinearSimple = self.clPrg.bilinear_simple
                        self.clKernBilinearLDS = self.clPrg.bilinear_lds
                        self.clKernBicubicSimple = self.clPrg.bicubic_simple
                        self.clKernBicubicLDS = self.clPrg.bicubic_lds
                        

    def bilinear(self, src, dst):
        w = src.shape[1]
        h = src.shape[0]
        wnew = dst.shape[1]
        hnew = dst.shape[0]
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)

        clSrcImg = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        clDstImg = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(wnew, hnew))
        
        self.clKernBilinearSimple.set_args(clSrcImg, clDstImg)
        ev0 = cl.enqueue_copy(self.clQueue, clSrcImg, src, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernBilinearSimple, (wnew, hnew), None, wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, dst, clDstImg, origin=(0, 0), region=(wnew, hnew), wait_for=[ev1])
        ev2.wait()
        return get_elapsed_ms([ev0, ev1, ev2])

    def bicubic(self, src, dst):
        w = src.shape[1]
        h = src.shape[0]
        wnew = dst.shape[1]
        hnew = dst.shape[0]
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)
        
        clSrcImg = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        clDstImg = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(wnew, hnew))
        
        self.clKernBicubicSimple.set_args(clSrcImg, clDstImg)
        ev0 = cl.enqueue_copy(self.clQueue, clSrcImg, src, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernBicubicSimple, (wnew, hnew), None, wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, dst, clDstImg, origin=(0, 0), region=(wnew, hnew), wait_for=[ev1])
        ev2.wait()
        return get_elapsed_ms([ev0, ev1, ev2])

    def bilinear_lds(self, src, dst):
        w = src.shape[1]
        h = src.shape[0]
        wnew = dst.shape[1]
        hnew = dst.shape[0]
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)

        clSrcImg = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        clDstImg = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(wnew, hnew))

        self.clKernBilinearLDS.set_args(clSrcImg, clDstImg)
        ev0 = cl.enqueue_copy(self.clQueue, clSrcImg, src, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernBilinearLDS, (wnew, hnew), (16, 16), wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, dst, clDstImg, origin=(0, 0), region=(wnew, hnew), wait_for=[ev1])
        ev2.wait()
        return get_elapsed_ms([ev0, ev1, ev2])

    def bicubic_lds(self, src, dst):
        w = src.shape[1]
        h = src.shape[0]
        wnew = dst.shape[1]
        hnew = dst.shape[0]
        mf = cl.mem_flags
        fmt = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)

        clSrcImg = cl.Image(self.clCtx, mf.READ_ONLY, fmt, shape=(w, h))
        clDstImg = cl.Image(self.clCtx, mf.WRITE_ONLY, fmt, shape=(wnew, hnew))

        self.clKernBicubicLDS.set_args(clSrcImg, clDstImg)
        ev0 = cl.enqueue_copy(self.clQueue, clSrcImg, src, origin=(0, 0), region=(w, h))
        ev1 = cl.enqueue_nd_range_kernel(self.clQueue, self.clKernBicubicLDS, (wnew, hnew), (16, 16), wait_for=[ev0])
        ev2 = cl.enqueue_copy(self.clQueue, dst, clDstImg, origin=(0, 0), region=(wnew, hnew), wait_for=[ev1])
        ev2.wait()
        return get_elapsed_ms([ev0, ev1, ev2])

if __name__ == '__main__':

    util = clUtility()

    bgr = cv2.imread('images/lenna.png')
    w = 1280
    h = 720
    wnew = 2 * w
    hnew = 2 * h
    bgr = cv2.resize(bgr, (w, h))
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

    x = list(range(w))
    y = list(range(h))
    xnew = np.linspace(0, w-1, wnew)
    ynew = np.linspace(0, h-1, hnew)

    fb = interp2d(x, y, bgra[:,:,0], kind='linear')
    fg = interp2d(x, y, bgra[:,:,1], kind='linear')
    fr = interp2d(x, y, bgra[:,:,2], kind='linear')
    bnew = fb(xnew, ynew)
    gnew = fg(xnew, ynew)
    rnew = fr(xnew, ynew)
    anew = np.ones(rnew.shape, rnew.dtype) * 255
    linear2d = np.stack((bnew, gnew, rnew, anew), axis=2).astype(np.uint8)
    cv2.imwrite('linear-2d.bmp', linear2d)

    count = 0
    loopcount = 20
    profiling = 0
    while count < loopcount:
        t1 = time.time()
        linearCv2 = cv2.resize(bgra, (wnew, hnew), interpolation=cv2.INTER_LINEAR)
        profiling += time.time() - t1
        count += 1
    psnrR = peak_signal_noise_ratio(linearCv2[:,:,2], linear2d[:,:,2])
    psnrG = peak_signal_noise_ratio(linearCv2[:,:,1], linear2d[:,:,1])
    psnrB = peak_signal_noise_ratio(linearCv2[:,:,0], linear2d[:,:,0])
    print('linear: CV2 took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}'.format(profiling*1000/count, psnrR,psnrG,psnrB))
    cv2.imwrite('linear-cv2.bmp', linearCv2)

    count = 0
    profiling = None
    elapsed = 0
    linearCl = np.zeros((hnew, wnew, bgra.shape[2]), dtype=np.uint8)
    while count < loopcount:
        t1 = time.time()
        elapsed_list = util.bilinear(bgra, linearCl)
        count += 1
        elapsed += time.time() - t1
        if profiling is None:
            profiling = elapsed_list
        else:
            for i in range(len(elapsed_list)):
                profiling[i] += elapsed_list[i]
    psnrR = peak_signal_noise_ratio(linearCl[:,:,2], linear2d[:,:,2])
    psnrG = peak_signal_noise_ratio(linearCl[:,:,1], linear2d[:,:,1])
    psnrB = peak_signal_noise_ratio(linearCl[:,:,0], linear2d[:,:,0])
    print('linear: CL-simple took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}, '.format(elapsed*1000/count,psnrR,psnrG,psnrB), end='')
    print('profiling: {:.3f} + {:.3f} + {:.3f} ms'.format(profiling[0]/count, profiling[1]/count, profiling[2]/count))
    cv2.imwrite('linear-simple.bmp', linearCl)

    count = 0
    profiling = None
    elapsed = 0
    linearLDS = np.zeros((hnew, wnew, bgra.shape[2]), dtype=np.uint8)
    while count < loopcount:
        t1 = time.time()
        elapsed_list = util.bilinear_lds(bgra, linearLDS)
        count += 1
        elapsed += time.time() - t1
        if profiling is None:
            profiling = elapsed_list
        else:
            for i in range(len(elapsed_list)):
                profiling[i] += elapsed_list[i]
    t2 = time.time()
    psnrR = peak_signal_noise_ratio(linearLDS[:,:,2], linear2d[:,:,2])
    psnrG = peak_signal_noise_ratio(linearLDS[:,:,1], linear2d[:,:,1])
    psnrB = peak_signal_noise_ratio(linearLDS[:,:,0], linear2d[:,:,0])
    print('linear: CL-LDS took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}, '.format(elapsed*1000/count,psnrR,psnrG,psnrB), end='')
    print('profiling: {:.3f} + {:.3f} + {:.3f} ms'.format(profiling[0]/count, profiling[1]/count, profiling[2]/count))
    cv2.imwrite('linear-lds.bmp', linearLDS)

    fb = interp2d(x, y, bgra[:,:,0], kind='cubic')
    fg = interp2d(x, y, bgra[:,:,1], kind='cubic')
    fr = interp2d(x, y, bgra[:,:,2], kind='cubic')
    bnew = fb(xnew, ynew)
    gnew = fg(xnew, ynew)
    rnew = fr(xnew, ynew)
    anew = np.ones(rnew.shape, rnew.dtype) * 255
    cubic2d = np.stack((bnew, gnew, rnew, anew), axis=2).astype(np.uint8)
    cv2.imwrite('cubic-2d.bmp', cubic2d)

    count = 0
    loopcount = 20
    profiling = 0
    while count < loopcount:
        t1 = time.time()
        cubicCv2 = cv2.resize(bgra, (wnew, hnew), interpolation=cv2.INTER_CUBIC)
        profiling += time.time() - t1
        count += 1
    psnrR = peak_signal_noise_ratio(cubicCv2[:,:,2], cubic2d[:,:,2])
    psnrG = peak_signal_noise_ratio(cubicCv2[:,:,1], cubic2d[:,:,1])
    psnrB = peak_signal_noise_ratio(cubicCv2[:,:,0], cubic2d[:,:,0])
    print('cubic: CV2 took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}'.format(profiling*1000/count, psnrR,psnrG,psnrB))
    cv2.imwrite('cubic-cv2.bmp', cubicCv2)

    count = 0
    profiling = None
    elapsed = 0
    cubicCl = np.zeros((hnew, wnew, bgra.shape[2]), dtype=np.uint8)
    while count < loopcount:
        t1 = time.time()
        elapsed_list = util.bicubic(bgra, cubicCl)
        count += 1
        elapsed += time.time() - t1
        if profiling is None:
            profiling = elapsed_list
        else:
            for i in range(len(elapsed_list)):
                profiling[i] += elapsed_list[i]
    psnrR = peak_signal_noise_ratio(cubicCl[:,:,2], cubic2d[:,:,2])
    psnrG = peak_signal_noise_ratio(cubicCl[:,:,1], cubic2d[:,:,1])
    psnrB = peak_signal_noise_ratio(cubicCl[:,:,0], cubic2d[:,:,0])
    print('cubic: CL-simple took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}, '.format(elapsed*1000/count,psnrR,psnrG,psnrB), end='')
    print('profiling: {:.3f} + {:.3f} + {:.3f} ms'.format(profiling[0]/count, profiling[1]/count, profiling[2]/count))
    cv2.imwrite('cubic-simple.bmp', cubicCl)

    count = 0
    profiling = None
    elapsed = 0
    cubicLDS = np.zeros((hnew, wnew, bgra.shape[2]), dtype=np.uint8)
    while count < loopcount:
        t1 = time.time()
        elapsed_list = util.bicubic_lds(bgra, cubicLDS)
        count += 1
        elapsed += time.time() - t1
        if profiling is None:
            profiling = elapsed_list
        else:
            for i in range(len(elapsed_list)):
                profiling[i] += elapsed_list[i]
    t2 = time.time()
    psnrR = peak_signal_noise_ratio(cubicLDS[:,:,2], cubic2d[:,:,2])
    psnrG = peak_signal_noise_ratio(cubicLDS[:,:,1], cubic2d[:,:,1])
    psnrB = peak_signal_noise_ratio(cubicLDS[:,:,0], cubic2d[:,:,0])
    print('cubic: CL-LDS took {:.3f} ms, PSNR: R:{:.3f} G:{:.3f} B:{:.3f}, '.format(elapsed*1000/count,psnrR,psnrG,psnrB), end='')
    print('profiling: {:.3f} + {:.3f} + {:.3f} ms'.format(profiling[0]/count, profiling[1]/count, profiling[2]/count))
    cv2.imwrite('cubic-lds.bmp', cubicLDS)


