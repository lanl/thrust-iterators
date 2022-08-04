\\ Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
\\ This program was produced under U.S. Government contract
\\ 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
\\ operated by Triad National Security, LLC for the U.S. Department of
\\ Energy/National Nuclear Security Administration. All rights in the
\\ program are reserved by Triad National Security, LLC, and the
\\ U.S. Department of Energy/National Nuclear Security
\\ Administration. The Government is granted for itself and others acting
\\ on its behalf a nonexclusive, paid-up, irrevocable worldwide license
\\ in this material to reproduce, prepare derivative works, distribute
\\ copies to the public, perform publicly and display publicly, and to
\\ permit others to do so.


#pragma once

template <typename T = double>
struct cd_apply_1d_cuda {

    static void diffusion_v1_res(const int& i0,
                                 const int& i1,
                                 const T& alpha,
                                 const T& beta,
                                 const T* dx,
                                 const int& agcw,
                                 const T* a,
                                 const int& ugcw,
                                 const T* u,
                                 const int& fgcw,
                                 const T* f,
                                 const T* f0,
                                 const int& rgcw,
                                 T* res);

    static void diffusion_v2_res(const int& i0,
                                 const int& i1,
                                 const T& alpha,
                                 const T& beta,
                                 const T* dx,
                                 const int& ugcw,
                                 const T* u,
                                 const int& fgcw,
                                 const T* f,
                                 const T* f0,
                                 const int& rgcw,
                                 T* res);

    static void poisson_v1_res(const int& i0,
                               const int& i1,
                               const T& beta,
                               const T* dx,
                               const int& fgcw,
                               const T* f,
                               const T* f0,
                               const int& rgcw,
                               T* res);

    static void diffusion_v1_apply(const int& i0,
                                   const int& i1,
                                   const T& alpha,
                                   const T& beta,
                                   const T* dx,
                                   const int& agcw,
                                   const T* a,
                                   const int& ugcw,
                                   const T* u,
                                   const T* f0,
                                   const int& rgcw,
                                   T* res);

    static void diffusion_v2_apply(const int& i0,
                                   const int& i1,
                                   const T& alpha,
                                   const T& beta,
                                   const T* dx,
                                   const int& ugcw,
                                   const T* u,
                                   const T* f0,
                                   const int& rgcw,
                                   T* res);

    static void poisson_v2_apply(const int& i0,
                                 const int& i1,
                                 const T& beta,
                                 const T* dx,
                                 const T* f0,
                                 const int& gcw,
                                 T* res);
};
