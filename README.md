# vkpeak

![CI](https://github.com/nihui/vkpeak/workflows/CI/badge.svg)
![download](https://img.shields.io/github/downloads/nihui/vkpeak/total.svg)

A synthetic benchmarking tool to measure peak capabilities of vulkan devices. It only measures the peak metrics that can be achieved using vector operations and does not represent a real-world use case.

## [Download](https://github.com/nihui/vkpeak/releases)

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia/Apple GPU

**https://github.com/nihui/vkpeak/releases**

## Usages

```shell
vkpeak.exe
```

vkpeak will choose the default vulkan device.

If you need to specify device id, then

```shell
vkpeak.exe 0
```

The only parameter 0 is the device id.

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Clone this project with all submodules

```shell
git clone https://github.com/nihui/vkpeak.git
cd vkpeak
git submodule update --init --recursive
```

2. Build with CMake
  - You can pass -DVulkan_LIBRARY=<path to your macOS/lib/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a> option to link static MoltenVK library on MacOS, MoltenVK is part of Vulkan SDK from https://vulkan.lunarg.com/

```shell
mkdir build
cd build
cmake ..
cmake --build . -j 4
```

## Sample

NVIDIA RTX5060Ti 16GB
```
device       = NVIDIA GeForce RTX 5060 Ti

fp32-scalar  = 17137.46 GFLOPS
fp32-vec4    = 16910.07 GFLOPS

fp16-scalar  = 12730.03 GFLOPS
fp16-vec4    = 12715.02 GFLOPS
fp16-matrix  = 101485.35 GFLOPS

fp64-scalar  = 398.59 GFLOPS
fp64-vec4    = 394.08 GFLOPS

int32-scalar = 12703.68 GIOPS
int32-vec4   = 12181.98 GIOPS

int16-scalar = 12690.05 GIOPS
int16-vec4   = 12208.29 GIOPS

int64-scalar = 3104.59 GIOPS
int64-vec4   = 2666.86 GIOPS

int8-dotprod = 16101.59 GIOPS
int8-matrix  = 202947.80 GIOPS

bf16-dotprod = 0.00 GFLOPS
bf16-matrix  = 0.00 GFLOPS

fp8-matrix   = 0.00 GFLOPS
bf8-matrix   = 0.00 GFLOPS

copy-h2h     = 18.17 GBPS
copy-h2d     = 17.93 GBPS
copy-d2h     = 18.09 GBPS
copy-d2d     = 190.70 GBPS
```

AMD RX9060XT 16GB
```
device       = AMD Radeon Graphics (RADV GFX1200)

fp32-scalar  = 17678.94 GFLOPS
fp32-vec4    = 12213.09 GFLOPS

fp16-scalar  = 16964.96 GFLOPS
fp16-vec4    = 22503.93 GFLOPS
fp16-matrix  = 107157.36 GFLOPS

fp64-scalar  = 444.89 GFLOPS
fp64-vec4    = 439.69 GFLOPS

int32-scalar = 2810.86 GIOPS
int32-vec4   = 2802.29 GIOPS

int16-scalar = 15055.07 GIOPS
int16-vec4   = 24225.60 GIOPS

int64-scalar = 937.59 GIOPS
int64-vec4   = 773.06 GIOPS

int8-dotprod = 54025.21 GIOPS
int8-matrix  = 194220.35 GIOPS

bf16-dotprod = 0.00 GFLOPS
bf16-matrix  = 0.00 GFLOPS

fp8-matrix   = 0.00 GFLOPS
bf8-matrix   = 0.00 GFLOPS

copy-h2h     = 19.73 GBPS
copy-h2d     = 19.32 GBPS
copy-d2h     = 24.28 GBPS
copy-d2d     = 145.94 GBPS
```

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
