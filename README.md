# vkpeak

![CI](https://github.com/nihui/vkpeak/workflows/CI/badge.svg)

A synthetic benchmarking tool to measure peak capabilities of vulkan devices. It only measures the peak metrics that can be achieved using vector operations and does not represent a real-world use case.

## [Download](https://github.com/nihui/vkpeak/releases)

Download Windows/Linux/MacOS Executable for Intel/AMD/Nvidia GPU

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

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/nihui/vkpeak.git
cd vkpeak
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

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

int32-scalar = 8363.63 GILOPS
int32-vec4   = 8313.07 GILOPS

int16-scalar = 5518.05 GILOPS
int16-vec4   = 7138.91 GILOPS
```

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
