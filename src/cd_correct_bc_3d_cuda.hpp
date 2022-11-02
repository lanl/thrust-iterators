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
struct cd_correct_bc_3d_cuda {
    static void set_bc(const int& i0,
                       const int& i1,
                       const int& j0,
                       const int& j1,
                       const int& k0,
                       const int& k1,
                       const T* dx,
                       const int& dgcw,
                       const T* d0,
                       const T* d1,
                       const T* d2,
                       const int& ugcw,
                       T* u,
                       const int* bLo,
                       const int* bHi,
                       const int& exOrder,
                       const int& face,
                       const int& type,
                       const int& btype,
                       const T& alpha,
                       const T& beta);

    static void set_poisson_bc(const int& i0,
                               const int& i1,
                               const int& j0,
                               const int& j1,
                               const int& k0,
                               const int& k1,
                               const T* dx,
                               const int& ugcw,
                               T* u,
                               const int* bLo,
                               const int* bHi,
                               const int& exOrder,
                               const int& face,
                               const int& type,
                               const int& btype,
                               const T& alpha,
                               const T& beta);
};
