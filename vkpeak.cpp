// vkpeak implemented with ncnn library

#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>

static const char glsl_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afp c0 = afp(gx);
    afp c1 = afp(lx);

    afp a = c0;
    afp b = c1;

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = sfp(c0);
}
)";

static const char glsl_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { float c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    afpvec4 c0 = afpvec4(gx);
    afpvec4 c1 = afpvec4(lx);

    afpvec4 a = c0 + afpvec4(0,1,2,-3);
    afpvec4 b = c1 + afpvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = sfp((c0[0] + c0[1]) + (c0[2] + c0[3]));
}
)";

static const char glsl_fp64_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    double c0 = double(gx);
    double c1 = double(lx);

    double a = c0;
    double b = c1;

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_fp64_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { double c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    dvec4 c0 = dvec4(gx);
    dvec4 c1 = dvec4(lx);

    dvec4 a = c0 + dvec4(0,1,2,-3);
    dvec4 b = c1 + dvec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_int32_p1_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int c0 = int(gx);
    int c1 = int(lx);

    int a = c0;
    int b = c1;

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_int32_p4_data[] = R"(
#version 450

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    ivec4 c0 = ivec4(gx);
    ivec4 c1 = ivec4(lx);

    ivec4 a = c0 + ivec4(0,1,2,-3);
    ivec4 b = c1 + ivec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_int16_p1_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int16_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    int16_t c0 = int16_t(gx);
    int16_t c1 = int16_t(lx);

    int16_t a = c0;
    int16_t b = c1;

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = c0;
}
)";

static const char glsl_int16_p4_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require

layout (constant_id = 0) const int loop = 1;

layout (binding = 0) writeonly buffer c_blob { int16_t c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    i16vec4 c0 = i16vec4(gx);
    i16vec4 c1 = i16vec4(lx);

    i16vec4 a = c0 + i16vec4(0,1,2,-3);
    i16vec4 b = c1 + i16vec4(2,3,5,-7);

    for (int i = 0; i < loop; i++)
    {
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
        c0 = a * c0 + b;
        c1 = a * c1 + b;
    }

    c0 = c0 + c1;
    c_blob_data[gx] = (c0[0] + c0[1]) + (c0[2] + c0[3]);
}
)";

static const char glsl_fp16_matrix_nv_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_NV_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { uvec4 c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    fcoopmatNV<16, gl_ScopeSubgroup, M, K> a = fcoopmatNV<16, gl_ScopeSubgroup, M, K>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, K, N> b = fcoopmatNV<16, gl_ScopeSubgroup, K, N>(float(lx));

    fcoopmatNV<16, gl_ScopeSubgroup, M, N> c0 = fcoopmatNV<16, gl_ScopeSubgroup, M, N>(float(gx));
    fcoopmatNV<16, gl_ScopeSubgroup, M, N> c1 = fcoopmatNV<16, gl_ScopeSubgroup, M, N>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
        c0 = coopMatMulAddNV(a, c0, b);
        c1 = coopMatMulAddNV(a, c1, b);
    }

    c0 = c0 + c1;
    coopMatStoreNV(c0, c_blob_data, gx, 0, false);
}
)";

static const char glsl_fp16_matrix_khr_data[] = R"(
#version 450

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_KHR_cooperative_matrix: require

layout (constant_id = 0) const int loop = 1;
layout (constant_id = 1) const int M = 1;
layout (constant_id = 2) const int N = 1;
layout (constant_id = 3) const int K = 1;

layout (binding = 0) writeonly buffer c_blob { uvec4 c_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint lx = gl_LocalInvocationID.x;

    coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>(float(lx));

    coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(gx));
    coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>(float(lx));

    for (int i = 0; i < loop; i++)
    {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
    }

    c0 = c0 + c1;
    coopMatStore(c0, c_blob_data, gx, 0, gl_CooperativeMatrixLayoutRowMajor);
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
    if (!vkdev->info.support_cooperative_matrix_16_16_16() && !vkdev->info.support_cooperative_matrix_8_8_16() && packing_type == 256)
    {
        return 0;
    }

    // check shader fp64 feature
    bool has_shader_fp64 = vkdev->info.physicalDevicefeatures().shaderFloat64;
    if (!has_shader_fp64 && (storage_type == 2 || arithmetic_type == 2))
    {
        return 0;
    }

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = storage_type == 1;
    opt.use_fp16_storage = storage_type == 1 || storage_type == 4;
    opt.use_fp16_arithmetic = arithmetic_type == 1;

    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    // reuse c storage, max 1G
    int buffer_size = std::min((int)(vkdev->get_heap_budget() / 8), 1 * 1024) * 1024 * 1024;
    if (vkdev->info.type() == 1)
    {
        // max 256M for integrated gpu
        buffer_size = std::min(buffer_size, 256 * 1024 * 1024);
    }
    ncnn::VkMat c(buffer_size, (size_t)1u, 1, allocator);

    int elemsize;
    if (storage_type == 0 || storage_type == 3)
    {
        // fp32 / int32
        elemsize = 4;
    }
    else if (storage_type == 1 || storage_type == 4)
    {
        // fp16 / int16
        elemsize = 2;
    }
    else if (storage_type == 2)
    {
        // fp64
        elemsize = 8;
    }
    else if (storage_type == 5)
    {
        // int8
        elemsize = 1;
    }

    int local_size_x = std::min(128, std::max(1, (int)vkdev->info.subgroup_size()));
    if (packing_type == 256)
    {
        // matrix on subgroup
        local_size_x = (int)vkdev->info.subgroup_size();
    }

    int M = 1;
    int N = 1;
    int K = 1;
    if (packing_type == 256)
    {
        if (vkdev->info.support_cooperative_matrix_16_16_16())
        {
            M = 16;
            N = 16;
            K = 16;
        }
        else if (vkdev->info.support_cooperative_matrix_16_8_16())
        {
            M = 16;
            N = 8;
            K = 16;
        }
        else if (vkdev->info.support_cooperative_matrix_8_8_16())
        {
            M = 8;
            N = 8;
            K = 16;
        }
        else if (vkdev->info.support_cooperative_matrix_16_8_8())
        {
            M = 16;
            N = 8;
            K = 8;
        }
    }

    int max_invocation_count = buffer_size / elemsize;
    if (packing_type == 256)
    {
        max_invocation_count = max_invocation_count / (M * N) * local_size_x;
    }

    double max_gflops = 0;

    // start with little works
    int invocation_count = max_invocation_count / 32;
    int loop = 16;

    bool rerun = true;

    // prepare storage
    while (rerun)
    {
        rerun = false;

        // setup pipeline
        ncnn::Pipeline pipeline(vkdev);
        {
            pipeline.set_local_size_xyz(local_size_x, 1, 1);

            std::vector<ncnn::vk_specialization_type> specializations(1);
            specializations[0].i = loop;

            // glsl to spirv
            // -1 for omit the tail '\0'
            std::vector<uint32_t> spirv;
            if (arithmetic_type == 2)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p1_data, sizeof(glsl_fp64_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_fp64_p4_data, sizeof(glsl_fp64_p4_data) - 1, opt, spirv);
                }
            }
            else if (arithmetic_type == 3)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int32_p1_data, sizeof(glsl_int32_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int32_p4_data, sizeof(glsl_int32_p4_data) - 1, opt, spirv);
                }
            }
            else if (arithmetic_type == 4)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_int16_p1_data, sizeof(glsl_int16_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_int16_p4_data, sizeof(glsl_int16_p4_data) - 1, opt, spirv);
                }
            }
            else // if (arithmetic_type == 0 || arithmetic_type == 1)
            {
                if (packing_type == 1)
                {
                    ncnn::compile_spirv_module(glsl_p1_data, sizeof(glsl_p1_data) - 1, opt, spirv);
                }
                if (packing_type == 4)
                {
                    ncnn::compile_spirv_module(glsl_p4_data, sizeof(glsl_p4_data) - 1, opt, spirv);
                }
                if (packing_type == 256)
                {
                    // loop M N K
                    specializations.resize(4);
                    specializations[1].i = M;
                    specializations[2].i = N;
                    specializations[3].i = K;

                    if (vkdev->info.support_VK_KHR_cooperative_matrix())
                    {
                        ncnn::compile_spirv_module(glsl_fp16_matrix_khr_data, sizeof(glsl_fp16_matrix_khr_data) - 1, opt, spirv);
                    }
                    else
                    {
                        ncnn::compile_spirv_module(glsl_fp16_matrix_nv_data, sizeof(glsl_fp16_matrix_nv_data) - 1, opt, spirv);
                    }
                }
            }

            pipeline.create(spirv.data(), spirv.size() * 4, specializations);
        }

        const int cmd_loop = 10;

        for (int i = 0; i < cmd_loop; i++)
        {
            // encode command
            ncnn::VkCompute cmd(vkdev);
            {
                std::vector<ncnn::VkMat> bindings(1);
                bindings[0] = c;

                std::vector<ncnn::vk_constant_type> constants(0);

                ncnn::VkMat dispatcher;
                dispatcher.w = invocation_count;
                dispatcher.h = 1;
                dispatcher.c = 1;
                cmd.record_pipeline(&pipeline, bindings, constants, dispatcher);
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

                if (time < 800)
                {
                    // for fast device
                    if (invocation_count < max_invocation_count)
                    {
                        invocation_count = std::min(invocation_count * 2, max_invocation_count);
                    }
                    else
                    {
                        loop *= 2;
                    }
                    rerun = true;
                    break;
                }

                double mac = (double)invocation_count * ((double)loop * 16 * 2 + 1); // +1 for the tail c0+c1

                if (packing_type == 256)
                {
                    mac *= (M * N) * 16;
                    mac /= local_size_x;
                }
                else
                {
                    mac *= packing_type;
                }

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
    // storage_type     = 0/1/2/3/4/5/6 = fp32 fp16 fp64 int32 int16 int8 bf16
    // arithmetic_type  = 0/1/2/3/4     = fp32 fp16 fp64 int32 int16
    // packing_type     = 1/4/256       = scalar vec4 matrix

    fprintf(stderr, "\n");
    fprintf(stderr, "fp32-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 0, 0, 1));
    fprintf(stderr, "fp32-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 0, 0, 4));

    fprintf(stderr, "\n");
    fprintf(stderr, "fp16-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 0, 1, 1));
    fprintf(stderr, "fp16-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 0, 1, 4));
    fprintf(stderr, "fp16-matrix  = %.2f GFLOPS\n", vkpeak(device_id, 1, 1, 256));

    fprintf(stderr, "\n");
    fprintf(stderr, "fp64-scalar  = %.2f GFLOPS\n", vkpeak(device_id, 2, 2, 1));
    fprintf(stderr, "fp64-vec4    = %.2f GFLOPS\n", vkpeak(device_id, 2, 2, 4));

    fprintf(stderr, "\n");
    fprintf(stderr, "int32-scalar = %.2f GIOPS\n", vkpeak(device_id, 3, 3, 1));
    fprintf(stderr, "int32-vec4   = %.2f GIOPS\n", vkpeak(device_id, 3, 3, 4));

    fprintf(stderr, "\n");
    fprintf(stderr, "int16-scalar = %.2f GIOPS\n", vkpeak(device_id, 4, 4, 1));
    fprintf(stderr, "int16-vec4   = %.2f GIOPS\n", vkpeak(device_id, 4, 4, 4));

    ncnn::destroy_gpu_instance();

    return 0;
}
