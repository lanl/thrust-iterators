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
