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

#include "cuda/traits.hpp"
template <typename T = double>
struct md_device_span_cuda {
    static void init(const int& i0,
                     const int& i1,
                     const T& beta,
                     const T* dx,
                     const int ugcw,
                     const T* u,
                     const int& rgcw,
                     T* res);

    static void stride(const int& i0,
                       const int& i1,
                       const int& stride,
                       const T& beta,
                       const T* dx,
                       const int ugcw,
                       const T* u,
                       const int& rgcw,
                       T* res);

    static void init(const int& i0,
                     const int& i1,
                     const int& j0,
                     const int& j1,
                     const T& beta,
                     const T* dx,
                     const int ugcw,
                     const T* u,
                     const int& rgcw,
                     T* res);

    static void stride(const int& i0,
                       const int& i1,
                       const int& is,
                       const int& j0,
                       const int& j1,
                       const int& js,
                       const T& beta,
                       const T* dx,
                       const int ugcw,
                       const T* u,
                       const int& rgcw,
                       T* res);

    static void stride(const int& i0,
                       const int& i1,
                       const int& is,
                       const int& j0,
                       const int& j1,
                       const int& js,
                       const int& k0,
                       const int& k1,
                       const int& ks,
                       const T& beta,
                       const T* dx,
                       const int ugcw,
                       const T* u,
                       const int& rgcw,
                       T* res);
};
