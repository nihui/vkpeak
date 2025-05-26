# vkpeak

![CI](https://github.com/nihui/vkpeak/workflows/CI/badge.svg)

A synthetic benchmarking tool to measure peak capabilities of vulkan devices. It only measures the peak metrics that can be achieved using vector operations and does not represent a real-world use case.

## [Download](https://github.com/nihui/vkpeak/releases)

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia/Apple GPU

**https://github.com/nihui/vkpeak/releases**

## Usages

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

```
[nihui@nihui-pc build]$ ./vkpeak 1
device       = NVIDIA GeForce RTX 3060

fp32-scalar  = 9362.42 GFLOPS
fp32-vec4    = 11124.36 GFLOPS

fp16-scalar  = 13938.83 GFLOPS
fp16-vec4    = 10362.61 GFLOPS
fp16-matrix  = 56050.63 GFLOPS

fp64-scalar  = 219.45 GFLOPS
fp64-vec4    = 218.21 GFLOPS

int32-scalar = 7023.06 GIOPS
int32-vec4   = 7020.55 GIOPS

int16-scalar = 5500.50 GIOPS
int16-vec4   = 5827.67 GIOPS

int8-dotprod = 6691.18 GIOPS
int8-matrix  = 110374.23 GIOPS

bf16-dotprod = 0.00 GFLOPS
bf16-matrix  = 0.00 GFLOPS
```

```
nihui@nihui-macbook-air build % ./vkpeak 0 
device       = Apple M1

fp32-scalar  = 2092.68 GFLOPS
fp32-vec4    = 2381.70 GFLOPS

fp16-scalar  = 2198.92 GFLOPS
fp16-vec4    = 2495.72 GFLOPS
fp16-matrix  = 0.00 GFLOPS

fp64-scalar  = 0.00 GFLOPS
fp64-vec4    = 0.00 GFLOPS

int32-scalar = 651.10 GIOPS
int32-vec4   = 651.95 GIOPS

int16-scalar = 647.64 GIOPS
int16-vec4   = 653.83 GIOPS

int8-dotprod = 8821.90 GIOPS
int8-matrix  = 0.00 GIOPS

bf16-dotprod = 0.00 GFLOPS
bf16-matrix  = 0.00 GFLOPS
```

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
