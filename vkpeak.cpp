// vkpeak implemented with ncnn library

#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>

static const char glsl_p1_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { sfp a_blob_data[]; };
layout (binding = 1) buffer b_blob { sfp b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfp c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    afp a = buffer_ld1(a_blob_data, gx);
    afp b = buffer_ld1(b_blob_data, gx);

    afp c = afp(1.f);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    buffer_st1(a_blob_data, gx, a);
    buffer_st1(b_blob_data, gx, b);
    buffer_st1(c_blob_data, gx, c);
}
)";

static const char glsl_p4_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { sfpvec4 a_blob_data[]; };
layout (binding = 1) buffer b_blob { sfpvec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfpvec4 c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    afpvec4 a = buffer_ld4(a_blob_data, gx);
    afpvec4 b = buffer_ld4(b_blob_data, gx);

    afpvec4 c = afpvec4(1.f);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    buffer_st4(a_blob_data, gx, a);
    buffer_st4(b_blob_data, gx, b);
    buffer_st4(c_blob_data, gx, c);
}
)";

static const char glsl_fp64_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { double a_blob_data[]; };
layout (binding = 1) buffer b_blob { double b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    double a = a_blob_data[gx];
    double b = b_blob_data[gx];

    double c = double(1.f);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static const char glsl_fp64_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { dvec4 a_blob_data[]; };
layout (binding = 1) buffer b_blob { dvec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { dvec4 c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    dvec4 a = a_blob_data[gx];
    dvec4 b = b_blob_data[gx];

    dvec4 c = dvec4(1.f);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static const char glsl_int32_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { int a_blob_data[]; };
layout (binding = 1) buffer b_blob { int b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    int a = a_blob_data[gx];
    int b = b_blob_data[gx];

    int c = int(1);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static const char glsl_int32_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { ivec4 a_blob_data[]; };
layout (binding = 1) buffer b_blob { ivec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { ivec4 c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    ivec4 a = a_blob_data[gx];
    ivec4 b = b_blob_data[gx];

    ivec4 c = ivec4(1);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static const char glsl_int16_p1_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { int16_t a_blob_data[]; };
layout (binding = 1) buffer b_blob { int16_t b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { int16_t c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    int16_t a = a_blob_data[gx];
    int16_t b = b_blob_data[gx];

    int16_t c = int16_t(1);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static const char glsl_int16_p4_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) buffer a_blob { i16vec4 a_blob_data[]; };
layout (binding = 1) buffer b_blob { i16vec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { i16vec4 c_blob_data[]; };

void main()
{
    const int gx = int(gl_GlobalInvocationID.x);

    i16vec4 a = a_blob_data[gx];
    i16vec4 b = b_blob_data[gx];

    i16vec4 c = i16vec4(1);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    a_blob_data[gx] = a;
    b_blob_data[gx] = b;
    c_blob_data[gx] = c;
}
)";

static double vkpeak(int device_id, int storage_type, int arithmetic_type, int packing_type)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device_id);

    if (!vkdev)
    {
        return 0;
    }

    if (!vkdev->info.support_fp16_storage() && storage_type == 1)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_storage() && storage_type == 4)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_arithmetic() && arithmetic_type == 1)
    {
        return 0;
    }
    if (!vkdev->info.support_fp16_arithmetic() && arithmetic_type == 4)
    {
        return 0;
    }

    // query shader fp64 feature
    bool has_shader_fp64 = false;
    {
        VkPhysicalDevice physicalDevice = vkdev->info.physical_device();

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(physicalDevice, &features);

        has_shader_fp64 = features.shaderFloat64;
    }

    if (!has_shader_fp64 && (storage_type == 2 || arithmetic_type == 2))
    {
        return 0;
    }

    double max_gflops = 0;

    int loop = 1024;
    int count = 256 * 1024;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = storage_type == 1;
    opt.use_fp16_storage = storage_type == 1 || storage_type == 4;
    opt.use_fp16_arithmetic = arithmetic_type == 1;

    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    bool rerun = true;

    // prepare storage
    while (rerun)
    {
        rerun = false;

        // setup pipeline
        ncnn::Pipeline pipeline(vkdev);
        {
            int local_size_x = std::min(128, std::max(1, (int)vkdev->info.subgroup_size()));

            pipeline.set_local_size_xyz(local_size_x, 1, 1);

            std::vector<ncnn::vk_specialization_type> specializations(1);
            specializations[0].i = loop;

            // glsl to spirv
            // -1 for omit the tail '\0'
            std::vector<uint32_t> spirv;
            if (storage_type == 2)
            {
                if (packing_type == 0)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p1_data, sizeof(glsl_fp64_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p4_data, sizeof(glsl_fp64_p4_data) - 1, opt, spirv);
                }
            }
            else if (storage_type == 3)
            {
                if (packing_type == 0)
                {
                    ncnn::compile_spirv_module(glsl_int32_p1_data, sizeof(glsl_int32_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int32_p4_data, sizeof(glsl_int32_p4_data) - 1, opt, spirv);
                }
            }
            else if (storage_type == 4)
            {
                if (packing_type == 0)
                {
                    ncnn::compile_spirv_module(glsl_int16_p1_data, sizeof(glsl_int16_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int16_p4_data, sizeof(glsl_int16_p4_data) - 1, opt, spirv);
                }
            }
            else
            {
                if (packing_type == 0)
                {
                    ncnn::compile_spirv_module(glsl_p1_data, sizeof(glsl_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_p4_data, sizeof(glsl_p4_data) - 1, opt, spirv);
                }
            }

            pipeline.create(spirv.data(), spirv.size() * 4, specializations);
        }

        const int elempack = packing_type == 0 ? 1 : 4;

        ncnn::VkMat a;
        ncnn::VkMat b;
        ncnn::VkMat c;
        {
            if (storage_type == 2)
            {
                a.create(count, (size_t)(8u * elempack), elempack, allocator);
                b.create(count, (size_t)(8u * elempack), elempack, allocator);
                c.create(count, (size_t)(8u * elempack), elempack, allocator);
            }
            else if (opt.use_fp16_packed || opt.use_fp16_storage || storage_type == 4)
            {
                a.create(count, (size_t)(2u * elempack), elempack, allocator);
                b.create(count, (size_t)(2u * elempack), elempack, allocator);
                c.create(count, (size_t)(2u * elempack), elempack, allocator);
            }
            else
            {
                a.create(count, (size_t)(4u * elempack), elempack, allocator);
                b.create(count, (size_t)(4u * elempack), elempack, allocator);
                c.create(count, (size_t)(4u * elempack), elempack, allocator);
            }
        }

        const int cmd_loop = 20;

        for (int i = 0; i < cmd_loop; i++)
        {
            // encode command
            ncnn::VkCompute cmd(vkdev);
            {
                std::vector<ncnn::VkMat> bindings(3);
                bindings[0] = a;
                bindings[1] = b;
                bindings[2] = c;

                std::vector<ncnn::vk_constant_type> constants(0);

                cmd.record_pipeline(&pipeline, bindings, constants, c);
            }

            // time this
            {
                double t0 = ncnn::get_current_time();

                int ret = cmd.submit_and_wait();
                if (ret != 0)
                {
                    vkdev->reclaim_blob_allocator(allocator);
                    return 0;
                }

                double time = ncnn::get_current_time() - t0;

                if (time < 500)
                {
                    // for fast device
                    loop *= 2;
                    count *= 2;
                    rerun = true;
                    break;
                }

                const double mac = (double)count * (double)loop * 16 * elempack * 2;

                double gflops = mac / time / 1000000;

                // fprintf(stderr, "%f gflops\n", gflops);

                if (gflops > max_gflops)
                    max_gflops = gflops;
            }
        }
    }

    vkdev->reclaim_blob_allocator(allocator);

    return max_gflops;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [device_id]\n", argv[0]);
        return -1;
    }

    ncnn::create_gpu_instance();

    const int gpu_count = ncnn::get_gpu_count();
    if (gpu_count == 0)
    {
        fprintf(stderr, "No vulkan device\n");
        return -1;
    }

    const int device_id = atoi(argv[1]);
    if (device_id < 0 || device_id >= gpu_count)
    {
        fprintf(stderr, "No vulkan device for %d\n", device_id);
        fprintf(stderr, "Available devices:\n");

        for (int i = 0; i < gpu_count; i++)
        {
            fprintf(stderr, "%d = %s\n", i, ncnn::get_gpu_info(i).device_name());
        }

        return -1;
    }

    fprintf(stderr, "device       = %s\n", ncnn::get_gpu_info(device_id).device_name());

    // device_id        = 0
    // storage_type     = 0/1/2/3/4 = fp32 fp16 fp64 int32 int16
    // arithmetic_type  = 0/1/2/3/4 = fp32 fp16 fp64 int32 int16
    // packing_type     = 0/1       = scalar vec4

    fprintf(stderr, "\n");
    fprintf(stderr, "fp32-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 0, 0, 0));
    fprintf(stderr, "fp32-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 0, 0, 1));

    fprintf(stderr, "\n");
    fprintf(stderr, "fp16-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 1, 1, 0));
    fprintf(stderr, "fp16-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 1, 1, 1));

    fprintf(stderr, "\n");
    fprintf(stderr, "fp64-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 2, 2, 0));
    fprintf(stderr, "fp64-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 2, 2, 1));

    fprintf(stderr, "\n");
    fprintf(stderr, "int32-scalar = %.2f GIOPS\n", vkpeak(device_id, 3, 3, 0));
    fprintf(stderr, "int32-vec4   = %.2f GIOPS\n", vkpeak(device_id, 3, 3, 1));

    fprintf(stderr, "\n");
    fprintf(stderr, "int16-scalar = %.2f GIOPS\n", vkpeak(device_id, 4, 4, 0));
    fprintf(stderr, "int16-vec4   = %.2f GIOPS\n", vkpeak(device_id, 4, 4, 1));

    ncnn::destroy_gpu_instance();

    return 0;
}
