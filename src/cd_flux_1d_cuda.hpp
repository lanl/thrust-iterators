// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#pragma once

template <typename T = double>
struct cdf_1d_cuda {
    static void flux(const int& ifirst0,
                     const int& ilast0,
                     const T* dx,
                     const T* b0,
                     const int& gcw,
                     const T* u,
                     T* flux0);

    static void poisson_flux(const int& ifirst0,
                             const int& ilast0,
                             const T* dx,
                             const int& gcw,
                             const T* u,
                             T* flux0);
};
