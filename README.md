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

fp32-scalar  = 17606.54 GFLOPS
fp32-vec4    = 12155.22 GFLOPS

fp16-scalar  = 16921.16 GFLOPS
fp16-vec4    = 27833.48 GFLOPS
fp16-matrix  = 105337.66 GFLOPS

fp64-scalar  = 442.80 GFLOPS
fp64-vec4    = 437.55 GFLOPS

int32-scalar = 2804.59 GIOPS
int32-vec4   = 2796.74 GIOPS

int16-scalar = 15034.62 GIOPS
int16-vec4   = 26356.38 GIOPS

int64-scalar = 932.14 GIOPS
int64-vec4   = 768.53 GIOPS

int8-dotprod = 53893.32 GIOPS
int8-matrix  = 194476.41 GIOPS

bf16-dotprod = 24427.68 GFLOPS
bf16-matrix  = 105099.82 GFLOPS

fp8-matrix   = 205061.72 GFLOPS
bf8-matrix   = 208234.02 GFLOPS

copy-h2h     = 21.05 GBPS
copy-h2d     = 21.17 GBPS
copy-d2h     = 23.70 GBPS
copy-d2d     = 145.23 GBPS
```

AMD RX9070XT 16GB
```
device       = AMD Radeon RX 9070 XT (RADV GFX1201)

fp32-scalar  = 35499.88 GFLOPS
fp32-vec4    = 24469.68 GFLOPS

fp16-scalar  = 34127.63 GFLOPS
fp16-vec4    = 55966.64 GFLOPS
fp16-matrix  = 214419.79 GFLOPS

fp64-scalar  = 876.26 GFLOPS
fp64-vec4    = 871.68 GFLOPS

int32-scalar = 5624.66 GIOPS
int32-vec4   = 5617.52 GIOPS

int16-scalar = 29966.15 GIOPS
int16-vec4   = 53031.47 GIOPS

int64-scalar = 1871.93 GIOPS
int64-vec4   = 1543.96 GIOPS

int8-dotprod = 107156.45 GIOPS
int8-matrix  = 384988.43 GIOPS

bf16-dotprod = 49309.31 GFLOPS
bf16-matrix  = 213406.13 GFLOPS

fp8-matrix   = 421175.51 GFLOPS
bf8-matrix   = 424792.17 GFLOPS

copy-h2h     = 27.06 GBPS
copy-h2d     = 26.25 GBPS
copy-d2h     = 28.06 GBPS
copy-d2d     = 305.30 GBPS
```

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
