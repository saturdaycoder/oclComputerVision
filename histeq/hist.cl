#if HIST_N==2
    #define ushortN ushort2
    #define uintN uint2
    #define vloadN vload2
    #define vstoreN(_v, _off, _p) \
        do { \
            _p[_off] = _v.s0; \
            _p[_off + 1] = _v.s1; \
        } while(0)
#elif HIST_N==4
    #define ushortN ushort4
    #define uintN uint4
    #define vloadN vload4
    #define vstoreN(_v, _off, _p) \
        do { \
            _p[_off] = _v.s0; \
            _p[_off + 1] = _v.s1; \
            _p[_off + 2] = _v.s2; \
            _p[_off + 3] = _v.s3; \
        } while(0)
#elif HIST_N==8
    #define ushortN ushort8
    #define uintN uint8
    #define vloadN vload8
    #define vstoreN(_v, _off, _p) \
        do { \
            _p[_off] = _v.s0; \
            _p[_off + 1] = _v.s1; \
            _p[_off + 2] = _v.s2; \
            _p[_off + 3] = _v.s3; \
            _p[_off + 4] = _v.s4; \
            _p[_off + 5] = _v.s5; \
            _p[_off + 6] = _v.s6; \
            _p[_off + 7] = _v.s7; \
        } while(0)
#endif

__kernel void hist(const __global uchar *pImgIn,
                    const int width,
                    const int height,
                    __global uint *pHistOut)
{
    int globaly = get_global_id(0);
    int globalx = get_global_id(1);
    int globalh = get_global_size(0);
    int globalw = get_global_size(1);
    int localy = get_local_id(0);
    int localx = get_local_id(1);
    int localh = get_local_size(0);
    int localw = get_local_size(1);
    int groupy = get_group_id(0);
    int groupx = get_group_id(1);
    int grouph = get_num_groups(0);
    int groupw = get_num_groups(1);

    /* up to 32KiB local mem on AMD gfx902 */
    /* add 1 pixel padding to mitigate bank conflict */
    __local ushort hist[HIST_THREAD_NUM][HIST_BINS+1];

    /* memset */
    #pragma unroll
    for (int i = 0; i < HIST_BINS; ++i) {
        hist[localy][i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* each thread calculates one line (pixel number == BINS) */
    #pragma unroll
    for (int i = 0; i < HIST_BINS; ++i) {
        int gray_y = groupy * HIST_THREAD_NUM + localy;
        int gray_x = groupx * HIST_BINS + localx;
        uchar gray = pImgIn[gray_y * width + gray_x + i];
        hist[localy][gray] ++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* move global output pointer */
    pHistOut += (groupw * groupy + groupx) * HIST_BINS;
    
    /* accumulate histogram for the whole patch */
    ushortN acc = 0;
    #pragma unroll
    for (int i = 0; i < HIST_THREAD_NUM; ++i) {
        __local ushort *plocal = &hist[i][localy * HIST_N];
        ushortN v = vloadN(0, plocal);
        acc += v;
    }
    vstoreN(acc, localy * HIST_N, pHistOut);
}

__kernel void histeq_global(const __global uchar *pImgIn,
                    const int width,
                    const int height,
                    __global uchar *pImgOut,
                    const __global uchar *pMapping)
{
    int globaly = get_global_id(0);
    int globalx = get_global_id(1);
    uchar v = pImgIn[globaly * width + globalx];
    pImgOut[globaly * width + globalx] = pMapping[v];
}

__kernel void histeq_local_block(const __global uchar *pImgIn,
                    const int width,
                    const int height,
                    __global uchar *pImgOut,
                    const __global float *pMappingGrid,
                    const int blockWidth,
                    const int blockHeight,
                    const int blockNumX,
                    const int blockNumY)
{
    int y = get_global_id(0);
    int x = get_global_id(1);

    int b00idx = (x - blockWidth/2) / blockWidth;
    int b00x = b00idx * blockWidth + blockWidth/2;
    int b00idy = (y - blockHeight/2) / blockHeight;
    int b00y = b00idy * blockHeight + blockHeight/2;
    int b01idx = b00idx + 1;
    int b01idy = b00idy;
    int b10idx = b00idx;
    int b10idy = b00idy + 1;

    if (b01idx >= blockNumX)
        b01idx = blockNumX - 1;
    if (b10idy >= blockNumY)
        b10idy = blockNumY - 1;

    int b11idx = b01idx;
    int b11idy = b10idy;

    float s = (x - b00x) / (float)blockWidth;
    float t = (y - b00y) / (float)blockHeight;

    s = clamp(s, 0.0f, 1.0f);
    t = clamp(t, 0.0f, 1.0f);

    const __global float *pF00 = pMappingGrid + (b00idy * blockNumX + b00idx) * HIST_BINS;
    const __global float *pF01 = pMappingGrid + (b01idy * blockNumX + b01idx) * HIST_BINS;
    const __global float *pF10 = pMappingGrid + (b10idy * blockNumX + b10idx) * HIST_BINS;
    const __global float *pF11 = pMappingGrid + (b11idy * blockNumX + b11idx) * HIST_BINS;

    uchar v = pImgIn[y * width + x];
    pImgOut[y * width + x] = clamp((1-s) * (1-t) * pF00[v] + s * (1-t) * pF01[v] + (1-s) * t * pF10[v] + s * t * pF11[v], 0.0f, 255.0f);
}