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
