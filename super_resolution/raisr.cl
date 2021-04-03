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

#ifndef FILTER_LEN
#define FILTER_LEN (11)
#endif

#ifndef WORK_GROUP_WIDTH
#define WORK_GROUP_WIDTH (16)
#endif

#ifndef WORK_GROUP_HEIGHT
#define WORK_GROUP_HEIGHT (16)
#endif

#define LOCAL_PATCH_WIDTH ((FILTER_LEN>>1<<1) + WORK_GROUP_WIDTH)
#define LOCAL_PATCH_HEIGHT ((FILTER_LEN>>1<<1) + WORK_GROUP_HEIGHT)
#define LOCAL_GRAD_WIDTH ((FILTER_LEN>>1<<1) + WORK_GROUP_WIDTH - 2)
#define LOCAL_GRAD_HEIGHT ((FILTER_LEN>>1<<1) + WORK_GROUP_HEIGHT - 2)
#define GAUSS_LEN (FILTER_LEN - 2)

#define IS_GRAY(chn_order) (chn_order == CLK_R)
#define IS_RGBA(chn_order) (chn_order == CLK_RGBA || chn_order == CLK_BGRA || chn_order == CLK_ARGB)

#define CONV3x3(patch, kern) \
        (dot(patch[0], kern[2].s210) \
        + dot(patch[1], kern[1].s210) \
        + dot(patch[2], kern[0].s210))

__inline half4 linear_sample(read_only image2d_t src,
                    float2 norm_coord)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    half4 neighbor_pix[2][2];
    int srcw = get_image_width(src);
    int srch = get_image_height(src);
    float2 coord_in = norm_coord * (float2)(srcw-1, srch-1);
    int x00 = floor(coord_in.x);
    int y00 = floor(coord_in.y);
    half u = coord_in.x - floor(coord_in.x);
    half v = coord_in.y - floor(coord_in.y);
    half4 pix00 = convert_half4(read_imagef(src, sampler, (int2)(x00, y00)));
    half4 pix01 = convert_half4(read_imagef(src, sampler, (int2)(x00+1, y00)));
    half4 pix10 = convert_half4(read_imagef(src, sampler, (int2)(x00, y00+1)));
    half4 pix11 = convert_half4(read_imagef(src, sampler, (int2)(x00+1, y00+1)));
    return (1-u)*(1-v)*pix00 + u*(1-v)*pix01 + (1-u)*v*pix10 + u*v*pix11;
}

__constant half4 cubic_matrix[4]={
    (half4)(0.0h, -0.5h, 1.0h, -0.5h),
    (half4)(1.0h, 0.0h, -2.5h, 1.5h),
    (half4)(0.0h, 0.5h, 2.0h, -1.5h),
    (half4)(0.0h, 0.0h, -0.5h, 0.5h)
};
__inline half4 cubic_sample(read_only image2d_t src,
                    float2 norm_coord)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    half4 neighbor_pix[4][4];
    int srcw = get_image_width(src);
    int srch = get_image_height(src);
    float2 coord_in = norm_coord * (float2)(srcw-1, srch-1);
    int x00 = floor(coord_in.x) - 1;
    int y00 = floor(coord_in.y) - 1;
    half4 pix_dst = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
    half u = coord_in.x - floor(coord_in.x);
    half u2 = u * u;
    half u3 = u2 * u;
    half4 us = (half4)(1, u, u2, u3);
    half xweight[4] = {
        dot(us, cubic_matrix[0]),
        dot(us, cubic_matrix[1]),
        dot(us, cubic_matrix[2]),
        dot(us, cubic_matrix[3])
    };
    half v = coord_in.y - floor(coord_in.y);
    half v2 = v * v;
    half v3 = v2 * v;
    half4 vs = (half4)(1, v, v2, v3);
    half yweight[4] = {
        dot(vs, cubic_matrix[0]),
        dot(vs, cubic_matrix[1]),
        dot(vs, cubic_matrix[2]),
        dot(vs, cubic_matrix[3])
    };
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            int2 sampler_coord = (int2)(x00 + j, y00 + i);
            half4 pix = convert_half4(read_imagef(src, sampler, sampler_coord));
            pix_dst += pix * xweight[j] * yweight[i];
        }
    }
    pix_dst = clamp(pix_dst, 0.0h, 1.0h);
    return pix_dst;
}

__kernel void raisr(read_only image2d_t src,
                    write_only image2d_t dst,
                    const __global float *grad_x_matrix,
                    const __global float *grad_y_matrix,
                    const __global float *csc_rgb2yuv_matrix,
                    const __global float *csc_yuv2rgb_matrix,
                    const __global float *gaussian_weights,
                    const __global float *strength_quantizers,
                    const __global float *coherence_quantizers,
                    const int scale_factor,
                    const __global float *filters)
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
    int srcchn = get_image_channel_order(src);
    int dstchn = get_image_channel_order(dst);
    
    __local half4 csc_rgb2yuv[4];
    __local half4 csc_yuv2rgb[4];
    __local half3 grad_kern_x[3];
    __local half3 grad_kern_y[3];
    
    __local half4 yuv_patch[LOCAL_PATCH_HEIGHT][LOCAL_PATCH_WIDTH];
    __local half2 grad[LOCAL_GRAD_HEIGHT][LOCAL_GRAD_WIDTH];
    __local half gaussian[GAUSS_LEN][GAUSS_LEN];

    if (localx == 0 && localy == 0) {
        if (IS_RGBA(srcchn)) {
            csc_rgb2yuv[0] = convert_half4(vload4(0, csc_rgb2yuv_matrix));
            csc_rgb2yuv[1] = convert_half4(vload4(1, csc_rgb2yuv_matrix));
            csc_rgb2yuv[2] = convert_half4(vload4(2, csc_rgb2yuv_matrix));
            csc_rgb2yuv[3] = convert_half4(vload4(3, csc_rgb2yuv_matrix));
        }
        if (IS_RGBA(dstchn)) {
            csc_yuv2rgb[0] = convert_half4(vload4(0, csc_yuv2rgb_matrix));
            csc_yuv2rgb[1] = convert_half4(vload4(1, csc_yuv2rgb_matrix));
            csc_yuv2rgb[2] = convert_half4(vload4(2, csc_yuv2rgb_matrix));
            csc_yuv2rgb[3] = convert_half4(vload4(3, csc_yuv2rgb_matrix));
        }
        grad_kern_x[0] = convert_half3(vload3(0, grad_x_matrix));
        grad_kern_x[1] = convert_half3(vload3(1, grad_x_matrix));
        grad_kern_x[2] = convert_half3(vload3(2, grad_x_matrix));
        grad_kern_y[0] = convert_half3(vload3(0, grad_y_matrix));
        grad_kern_y[1] = convert_half3(vload3(1, grad_y_matrix));
        grad_kern_y[2] = convert_half3(vload3(2, grad_y_matrix));
    }
    if (localy < GAUSS_LEN && localx < GAUSS_LEN) {
        gaussian[localy][localx] = gaussian_weights[localy * GAUSS_LEN + localx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    sampler_t direct_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int thread_offset = localy * WORK_GROUP_WIDTH + localx;
    int samp_per_thread = ceil((float)LOCAL_PATCH_HEIGHT * LOCAL_PATCH_WIDTH / WORK_GROUP_WIDTH / WORK_GROUP_HEIGHT);
    int grad_per_thread = ceil((float)LOCAL_GRAD_HEIGHT * LOCAL_GRAD_WIDTH / WORK_GROUP_WIDTH / WORK_GROUP_HEIGHT);
    int remaining_samps = min(LOCAL_PATCH_HEIGHT * LOCAL_PATCH_WIDTH - thread_offset * samp_per_thread, samp_per_thread);
    if (remaining_samps > 0) {
        int global_offx = localw * groupx - FILTER_LEN / 2;
        int global_offy = localh * groupy - FILTER_LEN / 2;
        int global_off1d = global_offy * dstw + global_offx;

        #pragma unroll
        for (int i = 0; i < remaining_samps; ++i) {
            int off1d = thread_offset * samp_per_thread + i;
            int patchx = off1d % LOCAL_PATCH_WIDTH;
            int patchy = off1d / LOCAL_PATCH_WIDTH;
            int sampx = patchx + global_offx;
            int sampy = patchy + global_offy;
            half4 pix_src;

            if (srcw == dstw && srch == dsth) {
                pix_src = convert_half4(read_imagef(src, direct_sampler, (int2)(sampx, sampy)));
            }
            else {
                float2 norm_coord = convert_float2((int2)(sampx, sampy)) / (float2)(dstw-1, dsth-1);
                pix_src = linear_sample(src, norm_coord);
            }

            if (IS_GRAY(srcchn)) {
                yuv_patch[patchy][patchx] = pix_src;
            }
            else if (IS_RGBA(srcchn)) {
                yuv_patch[patchy][patchx].x = dot(csc_rgb2yuv[0], pix_src);
                yuv_patch[patchy][patchx].y = dot(csc_rgb2yuv[1], pix_src);
                yuv_patch[patchy][patchx].z = dot(csc_rgb2yuv[2], pix_src);
                yuv_patch[patchy][patchx].w = dot(csc_rgb2yuv[3], pix_src);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int remaining_grads = min(LOCAL_GRAD_HEIGHT * LOCAL_GRAD_WIDTH - thread_offset * grad_per_thread, grad_per_thread);
    if (remaining_grads > 0) {
        #pragma unroll
        for (int i = 0; i < remaining_grads; ++i) {
            int off1d = thread_offset * grad_per_thread + i;
            int offx = off1d % LOCAL_GRAD_WIDTH;
            int offy = off1d / LOCAL_GRAD_WIDTH;
            half3 patch[3] = {
                (half3)(yuv_patch[offy][offx].x, yuv_patch[offy][offx+1].x, yuv_patch[offy][offx+2].x),
                (half3)(yuv_patch[offy+1][offx].x, yuv_patch[offy+1][offx+1].x, yuv_patch[offy+1][offx+2].x),
                (half3)(yuv_patch[offy+2][offx].x, yuv_patch[offy+2][offx+1].x, yuv_patch[offy+2][offx+2].x),
            };
            grad[offy][offx].x = CONV3x3(patch, grad_kern_x);
            grad[offy][offx].y = CONV3x3(patch, grad_kern_y);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    half ma = 0;
    half mb = 0;
    half mc = 0;
    half md = 0;

    #pragma unroll
    for (int i = 0; i < GAUSS_LEN; ++i) {
        #pragma unroll
        for (int j = 0; j < GAUSS_LEN; ++j) {
            int row = i + localy;
            int col = j + localx;
            half gx = grad[row][col].x;
            half gy = grad[row][col].y;
            ma += gx * gy * gaussian[j][i];
            mb += gx * gy * gaussian[j][i];
            md += gy * gy * gaussian[j][i];
        }
    }
    mc = mb;

    half T = ma + md;
    half D = ma * md - mb * mc;
    half sqrt_td = sqrt( (T * T)/4 - D );
    half L1 = T/2 + sqrt_td;
    half L2 = T/2 - sqrt_td;

    half theta = atan2(mb, L1 - md);
    if (theta < 0)
        theta += PI;

    half coherence = 0.0h;
    half2 l12 = (half2)(L1, L2);
    half2 sqrt_l12 = sqrt(l12);
    half sqrt_l1 = sqrt_l12.x;
    half sqrt_l2 = sqrt_l12.y;
    if (sqrt_l1 + sqrt_l2 != 0) {
        coherence = (sqrt_l1 - sqrt_l2) / (sqrt_l1 + sqrt_l2);
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

    half4 pix_dst = (half4)(0.0h, 0.0h, 0.0h, 0.0h);

    #if 0
    int hash = (((angle_idx * NUM_STRENGTH) * NUM_COHERENCE + coherence_idx) * num_pixel_type) + pixel_type;
    const __global float *pf = filters + hash * FILTER_LEN * FILTER_LEN;
    #else
    float pf[FILTER_LEN * FILTER_LEN] = { 0 };
    pf[FILTER_LEN * FILTER_LEN / 2] = 1;
    #endif

    #pragma unroll
    for (int i = 0; i < FILTER_LEN; ++i) {
        #pragma unroll
        for (int j = 0; j < FILTER_LEN; ++j) {
            pix_dst += yuv_patch[localy + i][localx + j] * (half)pf[i * FILTER_LEN + j];
        }
    }

    if (IS_GRAY(dstchn)) {
        write_imagef(dst, (int2)(dstx, dsty), convert_float4(pix_dst));
    }
    else if (IS_RGBA(dstchn)) {
        half4 rgba_dst;
        rgba_dst.x = dot(csc_yuv2rgb[0], pix_dst);
        rgba_dst.y = dot(csc_yuv2rgb[1], pix_dst);
        rgba_dst.z = dot(csc_yuv2rgb[2], pix_dst);
        rgba_dst.w = dot(csc_yuv2rgb[3], pix_dst);
        write_imagef(dst, (int2)(dstx, dsty), convert_float4(rgba_dst));
    }
}



