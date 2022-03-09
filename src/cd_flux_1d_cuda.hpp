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
