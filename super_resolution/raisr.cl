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

__constant half4 cubic_matrix[4]={
    (half4)(0, -1, 2, -1),
    (half4)(2, 0, -5, 3),
    (half4)(0, 1, 4, -3),
    (half4)(0, 0, -1, 1)
};

#define IS_GRAY(chn_order) (chn_order == CLK_R)
#define IS_RGBA(chn_order) (chn_order == CLK_RGBA || chn_order == CLK_BGRA || chn_order == CLK_ARGB)

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
#define LINEAR_SAMPLE(src, norm_coord) convert_half4(read_imagef(src, sampler, norm_coord))

#define CONV3x3(patch, kern) \
        (dot(patch[0], kern[2].s210) \
        + dot(patch[1], kern[1].s210) \
        + dot(patch[2], kern[0].s210))

__kernel void raisr(read_only image2d_t src,
                    write_only image2d_t dst,
                    const __global float *grad_x_matrix,
                    const __global float *grad_y_matrix,
                    const __global float *csc_rgba2yuva_matrix,
                    const __global float *csc_yuva2rgba_matrix,
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
    
    __local half4 csc_rgba2yuva[4];
    __local half4 csc_yuva2rgba[4];
    __local half3 grad_kern_x[3];
    __local half3 grad_kern_y[3];
    
    __local half y_patch[LOCAL_PATCH_HEIGHT][LOCAL_PATCH_WIDTH];
    __local half cb_patch[LOCAL_PATCH_HEIGHT][LOCAL_PATCH_WIDTH];
    __local half cr_patch[LOCAL_PATCH_HEIGHT][LOCAL_PATCH_WIDTH];
    __local half grad[LOCAL_GRAD_HEIGHT][LOCAL_GRAD_WIDTH][2];
    __local half gaussian[GAUSS_LEN][GAUSS_LEN];

    if (localx == 0 && localy == 0) {
        if (IS_RGBA(srcchn)) {
            csc_rgba2yuva[0] = convert_half4(vload4(0, csc_rgba2yuva_matrix));
            csc_rgba2yuva[1] = convert_half4(vload4(1, csc_rgba2yuva_matrix));
            csc_rgba2yuva[2] = convert_half4(vload4(2, csc_rgba2yuva_matrix));
            csc_rgba2yuva[3] = convert_half4(vload4(3, csc_rgba2yuva_matrix));
        }
        if (IS_RGBA(dstchn)) {
            csc_yuva2rgba[0] = convert_half4(vload4(0, csc_yuva2rgba_matrix));
            csc_yuva2rgba[1] = convert_half4(vload4(1, csc_yuva2rgba_matrix));
            csc_yuva2rgba[2] = convert_half4(vload4(2, csc_yuva2rgba_matrix));
            csc_yuva2rgba[3] = convert_half4(vload4(3, csc_yuva2rgba_matrix));
        }
        grad_kern_x[0] = convert_half3(vload3(0, grad_x_matrix));
        grad_kern_x[1] = convert_half3(vload3(1, grad_x_matrix));
        grad_kern_x[2] = convert_half3(vload3(2, grad_x_matrix));
        grad_kern_y[0] = convert_half3(vload3(0, grad_y_matrix));
        grad_kern_y[1] = convert_half3(vload3(1, grad_y_matrix));
        grad_kern_y[2] = convert_half3(vload3(2, grad_y_matrix));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localy < LOCAL_PATCH_HEIGHT/2 && localx < LOCAL_PATCH_WIDTH/2) {
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
                half4 pix_src = LINEAR_SAMPLE(src, norm_coord);

                if (IS_GRAY(srcchn)) {
                    y_patch[patchy][patchx] = pix_src.x;
                }
                else if (IS_RGBA(srcchn)) {
                    y_patch[patchy][patchx] = dot(csc_rgba2yuva[0], pix_src);
                    cb_patch[patchy][patchx] = dot(csc_rgba2yuva[1], pix_src);
                    cr_patch[patchy][patchx] = dot(csc_rgba2yuva[2], pix_src);
                }
            }
        }
    }

    if (localy < GAUSS_LEN && localx < GAUSS_LEN) {
        gaussian[localy][localx] = gaussian_weights[localy * GAUSS_LEN + localx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localy < LOCAL_GRAD_HEIGHT/2 && localx < LOCAL_GRAD_WIDTH/2) {
        half3 grad_patch[3] = {
            (half3)(y_patch[2*localy][2*localx], y_patch[2*localy][2*localx+1], y_patch[2*localy][2*localx+2]),
            (half3)(y_patch[2*localy+1][2*localx], y_patch[2*localy+1][2*localx+1], y_patch[2*localy+1][2*localx+2]),
            (half3)(y_patch[2*localy+2][2*localx], y_patch[2*localy+2][2*localx+1], y_patch[2*localy+2][2*localx+2]),
        };
        grad[2*localy][2*localx][0] = CONV3x3(grad_patch, grad_kern_x);
        grad[2*localy][2*localx][1] = CONV3x3(grad_patch, grad_kern_y);

        grad_patch[0] = (half3)(y_patch[2*localy][2*localx+1], y_patch[2*localy][2*localx+2], y_patch[2*localy][2*localx+3]);
        grad_patch[1] = (half3)(y_patch[2*localy+1][2*localx+1], y_patch[2*localy+1][2*localx+2], y_patch[2*localy+1][2*localx+3]);
        grad_patch[2] = (half3)(y_patch[2*localy+2][2*localx+1], y_patch[2*localy+2][2*localx+2], y_patch[2*localy+2][2*localx+3]);
        grad[2*localy][2*localx+1][0] = CONV3x3(grad_patch, grad_kern_x);
        grad[2*localy][2*localx+1][1] = CONV3x3(grad_patch, grad_kern_y);

        grad_patch[0] = (half3)(y_patch[2*localy+1][2*localx], y_patch[2*localy+1][2*localx+1], y_patch[2*localy+1][2*localx+2]);
        grad_patch[1] = (half3)(y_patch[2*localy+2][2*localx], y_patch[2*localy+2][2*localx+1], y_patch[2*localy+2][2*localx+2]);
        grad_patch[2] = (half3)(y_patch[2*localy+3][2*localx], y_patch[2*localy+3][2*localx+1], y_patch[2*localy+3][2*localx+2]);
        grad[2*localy+1][2*localx][0] = CONV3x3(grad_patch, grad_kern_x);
        grad[2*localy+1][2*localx][1] = CONV3x3(grad_patch, grad_kern_y);

        grad_patch[0] = (half3)(y_patch[2*localy+1][2*localx+1], y_patch[2*localy+1][2*localx+2], y_patch[2*localy+1][2*localx+3]);
        grad_patch[1] = (half3)(y_patch[2*localy+2][2*localx+1], y_patch[2*localy+2][2*localx+2], y_patch[2*localy+2][2*localx+3]);
        grad_patch[2] = (half3)(y_patch[2*localy+3][2*localx+1], y_patch[2*localy+3][2*localx+2], y_patch[2*localy+3][2*localx+3]);
        grad[2*localy+1][2*localx+1][0] = CONV3x3(grad_patch, grad_kern_x);
        grad[2*localy+1][2*localx+1][1] = CONV3x3(grad_patch, grad_kern_y);
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

    filters += angle_idx * NUM_STRENGTH * NUM_COHERENCE * num_pixel_type * FILTER_LEN * FILTER_LEN
                + strength_idx * NUM_COHERENCE * num_pixel_type * FILTER_LEN * FILTER_LEN
                + coherence_idx * num_pixel_type * FILTER_LEN * FILTER_LEN
                + pixel_type * FILTER_LEN * FILTER_LEN;
    #pragma unroll
    for (int i = 0; i < FILTER_LEN; ++i) {
        #pragma unroll
        for (int j = 0; j < FILTER_LEN; ++j) {
            pix_dst.x += y_patch[localy + i][localx + j] * filters[i * FILTER_LEN + j];
            if (IS_RGBA(dstchn)) {
                pix_dst.y += cb_patch[localy + i][localx + j] * filters[i * FILTER_LEN + j];
                pix_dst.z += cr_patch[localy + i][localx + j] * filters[i * FILTER_LEN + j];
            }
        }
    }

    pix_dst = clamp(pix_dst, 0.0h, 1.0h);
    
    if (IS_GRAY(dstchn)) {
        write_imagef(dst, (int2)(dstx, dsty), convert_float4(pix_dst));
    }
    else if (IS_RGBA(dstchn)) {
        half4 rgba_dst;
        rgba_dst.x = dot(csc_yuva2rgba[0], pix_dst);
        rgba_dst.y = dot(csc_yuva2rgba[1], pix_dst);
        rgba_dst.z = dot(csc_yuva2rgba[2], pix_dst);
        rgba_dst.w = 1.0h;
        write_imagef(dst, (int2)(dstx, dsty), convert_float4(rgba_dst));
    }
}



