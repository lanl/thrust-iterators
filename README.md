# amp_kernels

To build with thrust OpenMP backend configure cmake with `-D THRUST_DEVICE_SYSTEM=OMP`.  If you are building on a machine without CUDA support it is likely that Thrust is not installed in a system location visible to cmake. In that case, clone the repo:

```shell
git clone --recursive https://github.com/NVIDIA/thrust.git
```
and add `-D Thrust_DIR=<path-to-cloned-repo>/thrust/cmake` to your cmake flags
