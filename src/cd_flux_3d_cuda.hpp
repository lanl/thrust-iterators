#pragma once

template <typename T = double>
struct cdf_3d_cuda {
    static void flux(const int& i0,
                     const int& j0,
                     const int& k0,
                     const int& i1,
                     const int& j1,
                     const int& k1,
                     const T* dx,
                     const T* b0,
                     const T* b1,
                     const T* b2,
                     const int& gcw,
                     const T* u,
                     T* f0,
                     T* f1,
                     T* f2);

    static void poisson_flux(const int& i0,
                             const int& j0,
                             const int& k0,
                             const int& i1,
                             const int& j1,
                             const int& k1,
                             const T* dx,
                             const int& gcw,
                             const T* u,
                             T* f0,
                             T* f1,
                             T* f2);
};
