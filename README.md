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
[nihui@nihui-pc build]$ ./vkpeak 0
device       = GeForce RTX 2070

fp32-scalar  = 8536.18 GFLOPS
fp32-vec4    = 8473.82 GFLOPS

fp16-scalar  = 8405.30 GFLOPS
fp16-vec4    = 16261.30 GFLOPS

fp64-scalar  = 262.86 GFLOPS
fp64-vec4    = 262.86 GFLOPS

int32-scalar = 8363.63 GIOPS
int32-vec4   = 8313.07 GIOPS

int16-scalar = 5518.05 GIOPS
int16-vec4   = 7138.91 GIOPS
```

```
nihui@nihui-macbook-air vkpeak-20210424-macos % ./vkpeak 0 
device       = Apple M1

fp32-scalar  = 2093.55 GFLOPS
fp32-vec4    = 2369.02 GFLOPS

fp16-scalar  = 2195.79 GFLOPS
fp16-vec4    = 2513.04 GFLOPS

fp64-scalar  = 0.00 GFLOPS
fp64-vec4    = 0.00 GFLOPS

int32-scalar = 653.38 GIOPS
int32-vec4   = 649.56 GIOPS

int16-scalar = 653.42 GIOPS
int16-vec4   = 652.94 GIOPS
```

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
