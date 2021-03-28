#pragma OPENCL EXTENSION  cl_khr_fp16 : enable

#define PI M_PI

#ifndef NUM_ANGLE
#define NUM_ANGLE (24)
#endif

#ifndef NUM_STRENGTH
#define NUM_STRENGTH (3)
#endif

#ifndef NUM_COHERENCE
#define NUM_COHERENCE (3)
#endif

#ifndef NUM_PIXTYPE
#define NUM_PIXTYPE (4)
#endif

#ifndef FILTER_LEN
#define FILTER_LEN (11)
#endif

__constant half sobel_x[9] = {
    -1.0h,  0.0h,  1.0h,
    -2.0h,  0.0h,  2.0h,
    -1.0h,  0.0h,  1.0h,
};

__constant half sobel_y[9] = {
    -1.0h, -2.0h, -1.0h,
     0.0h,  0.0h,  0.0h,
     1.0h,  2.0h,  1.0h,
};

#if COLOR_BT601
__constant half4 csc_y = (half4)(0.299h, 0.587h, 0.114h, 0.0h);
__constant half4 csc_cb = (half4)(-0.14713h, -0.28886h, 0.436h, 0.5h);
__constant half4 csc_cr = (half4)(0.615h, -0.51499h, 0.10001h, 0.5h);
__constant half4 csc_r = (half4)(1.0h, 0.0h, 1.13983h, -1.13983h * 0.5h);
__constant half4 csc_g = (half4)(1.0h, -0.39465h, -0.58060h, (0.39465h + 0.58060h) * 0.5h);
__constant half4 csc_b = (half4)(1.0h, 2.03211h, 0.0h, -2.03211h * 0.5h);
#elif COLOR_BT709
__constant half4 csc_y = (half4)(0.2126h, 0.7152h, 0.0722h, 0.0h);
__constant half4 csc_cb = (half4)(-0.09991h, -0.33609h, 0.436h, 0.5h);
__constant half4 csc_cr = (half4)(0.615h, -0.55861h, 0.05639h, 0.5h);
__constant half4 csc_r = (half4)(1.0h, 0.0h, 1.28033h, -1.28033h * 0.5h);
__constant half4 csc_g = (half4)(1.0h, -0.21482h, -0.38059h, (0.21482h + 0.38059h) * 0.5h);
__constant half4 csc_b = (half4)(1.0h, 2.12798h, 0.0h, -2.12798h * 0.5h);
#endif

__inline half conv3x3(const __local half *patch,
              int patch_stride,
              __constant half *pkernel)
{
    int row, col;
    half out = 0.0;
    #pragma unroll
    for (row = 0; row < 3; ++row) {
        #pragma unroll
        for (col = 0; col < 3; ++col) {
            /* rotate the conv kernel by 180 degree */
            out += patch[row * patch_stride + col] * pkernel[(2-row) * 3 + (2-col)];
        }
    }
    return out;
}

__inline half4 sample_linear(read_only image2d_t src, float2 norm_coord)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    return convert_half4(read_imagef(src, sampler, norm_coord));
}

#define A (-0.5h)
__inline half8 cubic_weight_0_1(half8 x)
{
    return 1 - (A + 3) * pown(x, 2) + (A + 2) * pown(x, 3);
}
__inline half8 cubic_weight_1_2(half8 x)
{
    return -4 * A + 8 * A * x - 5 * A * pown(x, 2) + A * pown(x, 3);
}
__inline half16 get_cubic_weight(float2 coord)
{
    float2 uv = coord - floor(coord);
    float u = uv.x;
    float v = uv.y;
    half16 u16 = (half16)(
        1+u, u, 1-u, 2-u,
        1+u, u, 1-u, 2-u,
        1+u, u, 1-u, 2-u,
        1+u, u, 1-u, 2-u
    );
    half16 v16 = (half16)(
        1+v, 1+v, 1+v, 1+v,
        v, v, v, v,
        1-v, 1-v, 1-v, 1-v,
        2-v, 2-v, 2-v, 2-v
    );
    half16 wu;
    wu.s03478bcf = cubic_weight_1_2(u16.s03478bcf);
    wu.s12569ade = cubic_weight_0_1(u16.s12569ade);
    half16 wv;
    wv.s0123cdef = cubic_weight_1_2(v16.s0123cdef);
    wv.s456789ab = cubic_weight_0_1(v16.s456789ab);
    return wu * wv;
}
__inline half4 sample_cubic(read_only image2d_t src, float2 norm_coord)
{
    half4 pix[4][4];
    int srcw = get_image_width(src);
    int srch = get_image_height(src);
    float2 coord_in = norm_coord * (float2)(srcw-1.0f, srch-1.0f);
    int sampx00 = floor(coord_in.x) - 1;
    int sampy00 = floor(coord_in.y) - 1;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            pix[i][j] = convert_half4(read_imagef(src, sampler, (int2)(sampx00 + j, sampy00 + i)));
        }
    }
    half16 rs, gs, bs;
    rs = (half16)(
        pix[0][0].x, pix[0][1].x, pix[0][2].x, pix[0][3].x,
        pix[1][0].x, pix[1][1].x, pix[1][2].x, pix[1][3].x,
        pix[2][0].x, pix[2][1].x, pix[2][2].x, pix[2][3].x,
        pix[3][0].x, pix[3][1].x, pix[3][2].x, pix[3][3].x
    );
    gs = (half16)(
        pix[0][0].y, pix[0][1].y, pix[0][2].y, pix[0][3].y,
        pix[1][0].y, pix[1][1].y, pix[1][2].y, pix[1][3].y,
        pix[2][0].y, pix[2][1].y, pix[2][2].y, pix[2][3].y,
        pix[3][0].y, pix[3][1].y, pix[3][2].y, pix[3][3].y
    );
    bs = (half16)(
        pix[0][0].z, pix[0][1].z, pix[0][2].z, pix[0][3].z,
        pix[1][0].z, pix[1][1].z, pix[1][2].z, pix[1][3].z,
        pix[2][0].z, pix[2][1].z, pix[2][2].z, pix[2][3].z,
        pix[3][0].z, pix[3][1].z, pix[3][2].z, pix[3][3].z
    );
    half16 w = get_cubic_weight(coord_in);
    half4 pix_dst;
    pix_dst.x = dot(w.s0123, rs.s0123) + dot(w.s4567, rs.s4567) + dot(w.s89ab, rs.s89ab) + dot(w.scdef, rs.scdef);
    pix_dst.y = dot(w.s0123, gs.s0123) + dot(w.s4567, gs.s4567) + dot(w.s89ab, gs.s89ab) + dot(w.scdef, gs.scdef);
    pix_dst.z = dot(w.s0123, bs.s0123) + dot(w.s4567, bs.s4567) + dot(w.s89ab, bs.s89ab) + dot(w.scdef, bs.scdef);
    pix_dst.w = 1.0h;
    pix_dst = clamp(pix_dst, 0.0h, 1.0h);
    return pix_dst;
}

__kernel void raisr(read_only image2d_t src,
                    write_only image2d_t dst,
                    const int scale_factor,
                    const __global float gaussian_weights[],
                    const __global float strength_quantizers[],
                    const __global float coherence_quantizers[],
                    const __global float filters[NUM_ANGLE][NUM_STRENGTH][NUM_COHERENCE][NUM_PIXTYPE][FILTER_LEN * FILTER_LEN])
{
    int dstx = get_global_id(0);
    int dsty = get_global_id(1);
    int localx = get_local_id(0);
    int localy = get_local_id(1);
    int localw = get_local_size(0);
    int localh = get_local_size(1);
    int groupx = get_group_id(0);
    int groupy = get_group_id(1);
    int srcw = get_image_width(src);
    int srch = get_image_height(src);
    int dstw = get_image_width(dst);
    int dsth = get_image_height(dst);

    __local half y_patch[26][26];
    #if !IMAGE_GRAY
        __local half cb_patch[26][26];
        __local half cr_patch[26][26];
    #endif
    __local half grad[24][24][2];
    __local half gaussian[9][9];

    if (localy < 13 && localx < 13) {
        int samp_offx = localw * groupx - FILTER_LEN / 2;
        int samp_offy = localh * groupy - FILTER_LEN / 2;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int patchx = 2 * localx + j;
                int patchy = 2 * localy + i;
                int sampx = patchx + samp_offx;
                int sampy = patchy + samp_offy;
                float2 norm_coord = convert_float2((int2)(sampx, sampy)) / (float2)(dstw-1.0f, dsth-1.0f);
                half4 rgba_src = sample_linear(src, norm_coord);

                #if IMAGE_GRAY
                    y_patch[patchy][patchx] = rgba_src.x;
                #else
                    y_patch[patchy][patchx] = dot(csc_y, rgba_src);
                    cb_patch[patchy][patchx] = dot(csc_cb, rgba_src);
                    cr_patch[patchy][patchx] = dot(csc_cr, rgba_src);
                #endif
            }
        }
    }

    if (localy < 9 && localx < 9) {
        gaussian[localy][localx] = gaussian_weights[localy * 9 + localx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localy < 12 && localx < 12) {
        grad[2 * localy][2 * localx][0] = conv3x3(&y_patch[2 * localy][2 * localx], 26, sobel_x);
        grad[2 * localy][2 * localx][1] = conv3x3(&y_patch[2 * localy][2 * localx], 26, sobel_y);

        grad[2 * localy][2 * localx + 1][0] = conv3x3(&y_patch[2 * localy][2 * localx + 1], 26, sobel_x);
        grad[2 * localy][2 * localx + 1][1] = conv3x3(&y_patch[2 * localy][2 * localx + 1], 26, sobel_y);

        grad[2 * localy + 1][2 * localx][0] = conv3x3(&y_patch[2 * localy + 1][2 * localx], 26, sobel_x);
        grad[2 * localy + 1][2 * localx][1] = conv3x3(&y_patch[2 * localy + 1][2 * localx], 26, sobel_y);

        grad[2 * localy + 1][2 * localx + 1][0] = conv3x3(&y_patch[2 * localy + 1][2 * localx + 1], 26, sobel_x);
        grad[2 * localy + 1][2 * localx + 1][1] = conv3x3(&y_patch[2 * localy + 1][2 * localx + 1], 26, sobel_y);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    half ma = 0;
    half mb = 0;
    half mc = 0;
    half md = 0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            int row = i + localy;
            int col = j + localx;
            half gx = grad[row][col][0];
            half gy = grad[row][col][1];
            ma += gx * gy * gaussian[j][i];
            mb += gx * gy * gaussian[j][i];
            md += gy * gy * gaussian[j][i];
        }
    }
    mc = mb;

    half T = ma + md;
    half D = ma * md - mb * mc;
    half L1 = T/2 + sqrt( (T * T)/4 - D );
    half L2 = T/2 - sqrt( (T * T)/4 - D );

    half theta = atan2(mb, L1 - md);
    if (theta < 0)
        theta += PI;

    half coherence = 0.0f;
    if (sqrt(L1) + sqrt(L2) != 0) {
        coherence = ( sqrt(L1) - sqrt(L2) ) / ( sqrt(L1) + sqrt(L2) );
    }

    int angle_idx = theta / PI * NUM_ANGLE;
    angle_idx = clamp(angle_idx, 0, NUM_ANGLE-1);
    int num_pixel_type = scale_factor * scale_factor;
    int pixel_type = (dsty % scale_factor) * scale_factor  + (dstx % scale_factor);
    int strength_idx = NUM_STRENGTH - 1;
    for (int i = 0; i < NUM_STRENGTH - 1; ++i) {
        if (L1 < strength_quantizers[i]) {
            strength_idx = i;
            break;
        }
    }
    int coherence_idx = NUM_COHERENCE - 1;
    for (int i = 0; i < NUM_COHERENCE - 1; ++i) {
        if (L1 < coherence_quantizers[i]) {
            coherence_idx = i;
            break;
        }
    }

    half4 pix_dst = (half4)(0.0h, 0.0h, 0.0h, 1.0h);

    #pragma unroll
    for (int i = 0; i < FILTER_LEN; ++i) {
        #pragma unroll
        for (int j = 0; j < FILTER_LEN; ++j) {
            pix_dst.x += y_patch[localy + i][localx + j] * filters[angle_idx][strength_idx][coherence_idx][pixel_type][i * FILTER_LEN + j];
            #if !IMAGE_GRAY
                pix_dst.y += cb_patch[localy + i][localx + j] * filters[angle_idx][strength_idx][coherence_idx][pixel_type][i * FILTER_LEN + j];
                pix_dst.z += cr_patch[localy + i][localx + j] * filters[angle_idx][strength_idx][coherence_idx][pixel_type][i * FILTER_LEN + j];
            #endif
        }
    }

    //pix_dst.x = y_patch[localy + 5][localx + 5];
    //#if !IMAGE_GRAY
    //    pix_dst.y = cb_patch[localy + 5][localx + 5];
    //    pix_dst.z = cr_patch[localy + 5][localx + 5];
    //#endif

    pix_dst = clamp(pix_dst, 0.0f, 1.0f);
    
    #if IMAGE_GRAY
        write_imagef(dst, (int2)(dstx, dsty), pix_dst);
    #else
        half4 rgba_dst;
        rgba_dst.x = dot(csc_r, pix_dst);
        rgba_dst.y = dot(csc_g, pix_dst);
        rgba_dst.z = dot(csc_b, pix_dst);
        rgba_dst.w = 1.0f;
        write_imagef(dst, (int2)(dstx, dsty), convert_float4(rgba_dst));
    #endif
}



