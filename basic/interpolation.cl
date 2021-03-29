#pragma OPENCL EXTENSION  cl_khr_fp16 : enable

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} region_t;

__constant half4 cubic_matrix[4]={
    (half4)(0, -1, 2, -1),
    (half4)(2, 0, -5, 3),
    (half4)(0, 1, 4, -3),
    (half4)(0, 0, -1, 1)
};

__inline float2 get_norm_coord(int2 coord, int2 size)
{
    return convert_float2(coord) / (float2)(size.x-1, size.y-1);
}
__inline region_t get_unnorm_region(float2 coord, int2 size)
{
    region_t ret_coord;
    ret_coord.left = floor(coord.x * (size.x-1));
    ret_coord.top = floor(coord.y * (size.y-1));
    ret_coord.right = ceil(coord.x * (size.x-1));
    ret_coord.bottom = ceil(coord.y * (size.y-1));
    return ret_coord;
}

__kernel void bilinear_simple(read_only image2d_t src,
                            write_only image2d_t dst)
{
    int outx = get_global_id(0);
    int outy = get_global_id(1);
    int width_out = get_image_width(dst);
    int height_out = get_image_height(dst);
    int2 coord = (int2)(outx, outy);
    int2 size = (int2)(width_out, height_out);
    float2 norm_coord = get_norm_coord(coord, size);
    sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    float4 pix = read_imagef(src, sampler, norm_coord);
    write_imagef(dst, coord, pix);
}

__kernel void bicubic_simple(read_only image2d_t src,
                            write_only image2d_t dst)
{
    int outx = get_global_id(0);
    int outy = get_global_id(1);
    int width_in = get_image_width(src);
    int height_in = get_image_height(src);
    int width_out = get_image_width(dst);
    int height_out = get_image_height(dst);
    int2 coord = (int2)(outx, outy);
    int2 dst_size = (int2)(width_out, height_out);
    int2 src_size = (int2)(width_in, height_in);
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float2 norm_coord = get_norm_coord(coord, dst_size);
    float2 coord_in = norm_coord * (float2)(width_in-1.0f, height_in-1.0f);
    region_t region = get_unnorm_region(norm_coord, src_size);
    half4 pix_dst = (half4)(0.0h, 0.0h, 0.0h, 0.0h);

    half u = coord_in.x - floor(coord_in.x);
    half u2 = u * u;
    half u3 = u2 * u;
    half4 us = (half4)(1, u, u2, u3) * 0.5h;
    half xweight[4] = {
        dot(us, cubic_matrix[0]),
        dot(us, cubic_matrix[1]),
        dot(us, cubic_matrix[2]),
        dot(us, cubic_matrix[3])
    };
    half v = coord_in.y - floor(coord_in.y);
    half v2 = v * v;
    half v3 = v2 * v;
    half4 vs = (half4)(1, v, v2, v3) * 0.5h;
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
            int2 sampler_coord = (int2)(region.left - 1 + j, region.top - 1 + i);
            half4 pix = convert_half4(read_imagef(src, sampler, sampler_coord));
            pix_dst += pix * xweight[j] * yweight[i];
        }
    }
    pix_dst = clamp(pix_dst, 0.0h, 1.0h);
    write_imagef(dst, coord, convert_float4(pix_dst));
}

__kernel void bilinear_lds(read_only image2d_t src,
                        write_only image2d_t dst)
{
    int outx = get_global_id(0);
    int outy = get_global_id(1);
    int localx = get_local_id(0);
    int localy = get_local_id(1);
    int localw = get_local_size(0);
    int localh = get_local_size(1);
    int groupx = get_group_id(0);
    int groupy = get_group_id(1);
    int width_in = get_image_width(src);
    int height_in = get_image_height(src);
    int width_out = get_image_width(dst);
    int height_out = get_image_height(dst);
    int2 dst_size = (int2)(width_out, height_out);
    int2 src_size = (int2)(width_in, height_in);
    int xstart_in, xend_in;
    int ystart_in, yend_in;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    __local half4 local_pix[20][20];

    /* load pixels into local memory */
    {
        int2 coord_lt = (int2)(groupx * localw, groupy * localh);
        region_t region_lt_in = get_unnorm_region(get_norm_coord(coord_lt, dst_size), src_size);
        int2 coord_rb = (int2)((groupx + 1) * localw - 1, (groupy + 1) * localh - 1);
        region_t region_rb_in = get_unnorm_region(get_norm_coord(coord_rb, dst_size), src_size);
        xstart_in = region_lt_in.left;
        ystart_in = region_lt_in.top;
        xend_in = region_rb_in.right;
        yend_in = region_rb_in.bottom;

        if (localx <= xend_in - xstart_in && localy <= yend_in - ystart_in) {
            int2 sampler_coord = (int2)(xstart_in + localx, ystart_in + localy);
            local_pix[localy][localx] = convert_half4(read_imagef(src, sampler, sampler_coord));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int2 coord = (int2)(outx, outy);
    float2 norm_coord = get_norm_coord(coord, dst_size);
    region_t region_in = get_unnorm_region(norm_coord, src_size);
    float2 coord_in = norm_coord * convert_float2(src_size);
    half u = coord_in.x - region_in.left;
    half v = coord_in.y - region_in.top;

    half4 pix00 = local_pix[region_in.top - ystart_in][region_in.left - xstart_in];
    half4 pix01 = local_pix[region_in.top - ystart_in][region_in.right - xstart_in];
    half4 pix10 = local_pix[region_in.bottom - ystart_in][region_in.left - xstart_in];
    half4 pix11 = local_pix[region_in.bottom - ystart_in][region_in.right - xstart_in];
    half4 pixout = (1-u)*(1-v)*pix00 + u*(1-v)*pix01 + (1-u)*v*pix10 + u*v*pix11;
    write_imagef(dst, coord, convert_float4(pixout));
}

__kernel void bicubic_lds(read_only image2d_t src,
                        write_only image2d_t dst)
{
    int outx = get_global_id(0);
    int outy = get_global_id(1);
    int localx = get_local_id(0);
    int localy = get_local_id(1);
    int localw = get_local_size(0);
    int localh = get_local_size(1);
    int groupx = get_group_id(0);
    int groupy = get_group_id(1);
    int width_in = get_image_width(src);
    int height_in = get_image_height(src);
    int width_out = get_image_width(dst);
    int height_out = get_image_height(dst);
    int2 dst_size = (int2)(width_out, height_out);
    int2 src_size = (int2)(width_in, height_in);
    int xstart_in, xend_in;
    int ystart_in, yend_in;

    __local half4 l_pix[20][20];
    __local half l_xweight[16][4];
    __local half l_yweight[16][4];

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    {
        /* load pixels into local memory */
        int2 coord_lt = (int2)(groupx * localw, groupy * localh);
        region_t region_lt_in = get_unnorm_region(get_norm_coord(coord_lt, dst_size), src_size);
        int2 coord_rb = (int2)((groupx + 1) * localw - 1, (groupy + 1) * localh - 1);
        region_t region_rb_in = get_unnorm_region(get_norm_coord(coord_rb, dst_size), src_size);
        xstart_in = region_lt_in.left-1;
        ystart_in = region_lt_in.top-1;
        xend_in = region_rb_in.right+1;
        yend_in = region_rb_in.bottom+1;

        if (localx <= xend_in - xstart_in && localy <= yend_in - ystart_in) {
            int2 sampler_coord = (int2)(xstart_in + localx, ystart_in + localy);
            l_pix[localy][localx] = convert_half4(read_imagef(src, sampler, sampler_coord));
        }

        /* calculate xweight into local memory */
        if (localy < 4) {
            int coordx = groupx * localw + localx;
            half coordx_in = (half)coordx / (width_out - 1) * (width_in - 1);
            half u = coordx_in - floor(coordx_in);
            half u2 = u * u;
            half u3 = u2 * u;
            half4 us = (half4)(1, u, u2, u3) * 0.5h;
            l_xweight[localx][localy] = dot(us, cubic_matrix[localy]);
        }

        /* calculate yweight into local memory */
        if (localx < 4) {
            int coordy = groupy * localh + localy;
            half coordy_in = (half)coordy / (height_out - 1) * (height_in - 1);
            half v = coordy_in - floor(coordy_in);
            half v2 = v * v;
            half v3 = v2 * v;
            half4 vs = (half4)(1, v, v2, v3) * 0.5h;
            l_yweight[localy][localx] = dot(vs, cubic_matrix[localx]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord = (int2)(outx, outy);
    float2 norm_coord = get_norm_coord(coord, dst_size);
    region_t region_in = get_unnorm_region(norm_coord, src_size);
    float2 coord_in = norm_coord * (float2)(width_in-1.0f, height_in-1.0f);
    int xoff = region_in.left - 1 - xstart_in;
    int yoff = region_in.top - 1 - ystart_in;

    half4 pix_dst = (half4)(0.0h, 0.0h, 0.0h, 0.0h);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pix_dst += l_pix[yoff+i][xoff+j] * l_xweight[localx][j] * l_yweight[localy][i];
        }
    }
    pix_dst = clamp(pix_dst, 0.0h, 1.0h);
    write_imagef(dst, coord, convert_float4(pix_dst));
}
