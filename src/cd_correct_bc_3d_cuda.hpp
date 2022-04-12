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
