#pragma once

template <typename T = double>
struct cd_correct_bc_2d_cuda {
    static void set_bc(const int& i0,
                       const int& i1,
                       const int& j0,
                       const int& j1,
                       const T* dx,
                       const int& dgcw,
                       const T* d0,
                       const T* d1,
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

    static void set_corner_bc(const int& i0,
                              const int& i1,
                              const int& j0,
                              const int& j1,
                              const int& gcw,
                              const T* dx,
                              const T* d0,
                              const T* d1,
                              T* u,
                              const int* bLo,
                              const int* bHi,
                              const int& exOrder,
                              const int& face,
                              const int& type,
                              const int& btype);

    static void set_homogenous_bc(const int& i0,
                                  const int& j0,
                                  const int& i1,
                                  const int& j1,
                                  const int& face,
                                  const int* bLo,
                                  const int* bHi,
                                  const int& exOrder,
                                  T* u);
};
