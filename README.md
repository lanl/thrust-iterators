# amp_kernels

To build with thrust OpenMP backend configure cmake with `-D THRUST_DEVICE_SYSTEM=OMP`.  If you are building on a machine without CUDA support it is likely that Thrust is not installed in a system location visible to cmake. In that case, clone the repo:

```shell
git clone --recursive https://github.com/NVIDIA/thrust.git
```
and add `-D Thrust_DIR=<path-to-cloned-repo>/thrust/cmake` to your cmake flags

## cuda-clang
nvcc/nvc++ currently error out on this branch.  Clang can be used instead as the cuda compiler (as of cmake 3.19).  The following works on my system (with thrust installed in a system path)

```bash
cmake -H. -Bbuild \
      -D CMAKE_C_COMPILER=clang \
      -D CMAKE_CXX_COMPILER=clang++ \
      -D CMAKE_CUDA_COMPILER=clang++ \
      "$@"
```
