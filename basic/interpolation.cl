#pragma OPENCL EXTENSION  cl_khr_fp16 : enable

__constant sampler_t norm_linear_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t fixed_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} region_t;
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
    float4 pix = read_imagef(src, norm_linear_sampler, norm_coord);
    write_imagef(dst, coord, pix);
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
    half4 pix[4][4];
    float2 norm_coord = get_norm_coord(coord, dst_size);
    float2 coord_in = norm_coord * (float2)(width_in-1.0f, height_in-1.0f);
    region_t region = get_unnorm_region(norm_coord, src_size);

    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        #pragma unroll
        for (int j = 0; j < 4; ++ j) {
            int2 sampler_coord = (int2)(region.left - 1 + j, region.top - 1 + i);
            pix[i][j] = convert_half4(read_imagef(src, fixed_sampler, sampler_coord));
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

    __local float4 local_pix[MAX_LOCAL_HEIGHT][MAX_LOCAL_WIDTH];

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
            local_pix[localy][localx] = read_imagef(src, fixed_sampler, sampler_coord);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int2 coord = (int2)(outx, outy);
    float2 norm_coord = get_norm_coord(coord, dst_size);
    region_t region_in = get_unnorm_region(norm_coord, src_size);
    float2 coord_in = norm_coord * convert_float2(src_size);
    float u = coord_in.x - region_in.left;
    float v = coord_in.y - region_in.top;
    float4 pix00 = local_pix[region_in.top - ystart_in][region_in.left - xstart_in];
    float4 pix01 = local_pix[region_in.top - ystart_in][region_in.right - xstart_in];
    float4 pix10 = local_pix[region_in.bottom - ystart_in][region_in.left - xstart_in];
    float4 pix11 = local_pix[region_in.bottom - ystart_in][region_in.right - xstart_in];
    float4 pixout = (1-u)*(1-v)*pix00 + (1-u)*v*pix10 + u*(1-v)*pix01 + u*v*pix11;
    write_imagef(dst, coord, pixout);
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

    __local half4 local_pix[MAX_LOCAL_HEIGHT][MAX_LOCAL_WIDTH];

    /* load pixels into local memory */
    {
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
            local_pix[localy][localx] = convert_half4(read_imagef(src, fixed_sampler, sampler_coord));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int2 coord = (int2)(outx, outy);
    float2 norm_coord = get_norm_coord(coord, dst_size);
    region_t region_in = get_unnorm_region(norm_coord, src_size);
    float2 coord_in = norm_coord * (float2)(width_in-1.0f, height_in-1.0f);
    half16 w = get_cubic_weight(coord_in);
    int xoff = region_in.left - 1 - xstart_in;
    int yoff = region_in.top - 1 - ystart_in;

    half16 rs, gs, bs;
    rs = (half16)(
        local_pix[yoff+0][xoff+0].x, local_pix[yoff+0][xoff+1].x, local_pix[yoff+0][xoff+2].x, local_pix[yoff+0][xoff+3].x,
        local_pix[yoff+1][xoff+0].x, local_pix[yoff+1][xoff+1].x, local_pix[yoff+1][xoff+2].x, local_pix[yoff+1][xoff+3].x,
        local_pix[yoff+2][xoff+0].x, local_pix[yoff+2][xoff+1].x, local_pix[yoff+2][xoff+2].x, local_pix[yoff+2][xoff+3].x,
        local_pix[yoff+3][xoff+0].x, local_pix[yoff+3][xoff+1].x, local_pix[yoff+3][xoff+2].x, local_pix[yoff+3][xoff+3].x
    );
    gs = (half16)(
        local_pix[yoff+0][xoff+0].y, local_pix[yoff+0][xoff+1].y, local_pix[yoff+0][xoff+2].y, local_pix[yoff+0][xoff+3].y,
        local_pix[yoff+1][xoff+0].y, local_pix[yoff+1][xoff+1].y, local_pix[yoff+1][xoff+2].y, local_pix[yoff+1][xoff+3].y,
        local_pix[yoff+2][xoff+0].y, local_pix[yoff+2][xoff+1].y, local_pix[yoff+2][xoff+2].y, local_pix[yoff+2][xoff+3].y,
        local_pix[yoff+3][xoff+0].y, local_pix[yoff+3][xoff+1].y, local_pix[yoff+3][xoff+2].y, local_pix[yoff+3][xoff+3].y
    );
    bs = (half16)(
        local_pix[yoff+0][xoff+0].z, local_pix[yoff+0][xoff+1].z, local_pix[yoff+0][xoff+2].z, local_pix[yoff+0][xoff+3].z,
        local_pix[yoff+1][xoff+0].z, local_pix[yoff+1][xoff+1].z, local_pix[yoff+1][xoff+2].z, local_pix[yoff+1][xoff+3].z,
        local_pix[yoff+2][xoff+0].z, local_pix[yoff+2][xoff+1].z, local_pix[yoff+2][xoff+2].z, local_pix[yoff+2][xoff+3].z,
        local_pix[yoff+3][xoff+0].z, local_pix[yoff+3][xoff+1].z, local_pix[yoff+3][xoff+2].z, local_pix[yoff+3][xoff+3].z
    );
    half4 pix_dst;
    pix_dst.x = dot(w.s0123, rs.s0123) + dot(w.s4567, rs.s4567) + dot(w.s89ab, rs.s89ab) + dot(w.scdef, rs.scdef);
    pix_dst.y = dot(w.s0123, gs.s0123) + dot(w.s4567, gs.s4567) + dot(w.s89ab, gs.s89ab) + dot(w.scdef, gs.scdef);
    pix_dst.z = dot(w.s0123, bs.s0123) + dot(w.s4567, bs.s4567) + dot(w.s89ab, bs.s89ab) + dot(w.scdef, bs.scdef);
    pix_dst.w = 1.0h;
    pix_dst = clamp(pix_dst, 0.0h, 1.0h);

    write_imagef(dst, coord, convert_float4(pix_dst));
}
