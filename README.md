# thrust iterators

This library provides a simple domain specific language on top of
thrust iterators.  The primary goal is to replace the fortran kernels
in the [Advanced Multi-Physics
package](https://bitbucket.org/AdvancedMultiPhysics/amp/) with kernels
that can target heterogenous architectures.  The use of the iterators
are demonstrated in the various tests that compare the output to those
of the fortran kernels.

# dependencies

Building requires [Catch2](https://github.com/catchorg/Catch2) (v3), [Boost](https://www.boost.org/) (for mpl), [thrust](https://github.com/NVIDIA/thrust) and a c++ 17 compiler

To build with thrust OpenMP backend configure cmake with `-D THRUST_DEVICE_SYSTEM=OMP`.  If you are building on a machine without CUDA support it is likely that Thrust is not installed in a system location visible to cmake. In that case, clone the repo:

```shell
git clone --recursive https://github.com/NVIDIA/thrust.git
```
and add `-D Thrust_DIR=<path-to-cloned-repo>/thrust/cmake` to your cmake flags

# license and copyright

With the exception of the  `m4` files, this project is copyright (c) 2022 Triad National Security, LLC, all rights reserved, and released under the [BSD-3 License](https://opensource.org/licenses/BSD-3-Clause) via LANL open-source copyright assertion C22058.

The `m4` files are copyright and licensed as indicated using what appears to be the [ISC License](https://opensource.org/licenses/ISC)
