#pragma once

template <typename T = double>
struct cd_correct_bc_1d_cuda {
    static void set_bc(const int& i0,
                       const int& i1,
                       const T* dx,
                       const int& dgcw,
                       const T* d0,
                       const int& ugcw,
                       T* u,
                       const int* bLo,
                       const int* bHi,
                       const int& exOrder,
                       const int& face,
                       const int& btype,
                       const T& alpha,
                       const T& beta);

    static void set_poisson_bc(const int& i0,
                               const int& i1,
                               const T* dx,
                               const int& ugcw,
                               T* u,
                               const int* bLo,
                               const int* bHi,
                               const int& exOrder,
                               const int& face,
                               const int& btype,
                               const T& alpha,
                               const T& beta);
};
