#pragma once

template <typename T = double>
struct cdf_2d_cuda {
    static void flux(const int& ifirst0,
                     const int& ifirst1,
                     const int& ilast0,
                     const int& ilast1,
                     const T* dx,
                     const T* b0,
                     const T* b1,
                     const int& gcw,
                     const T* u,
                     T* flux0,
                     T* flux1);

    static void poisson_flux(const int& ifirst0,
                             const int& ifirst1,
                             const int& ilast0,
                             const int& ilast1,
                             const T* dx,
                             const int& gcw,
                             const T* u,
                             T* flux0,
                             T* flux1);
};
