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
struct cd_stencil_1d_cuda {
    static void offdiag(const int& i0,
                        const int& i1,
                        const int& bi0,
                        const int& bi1,
                        const T* dx,
                        const T& beta,
                        const T* b0,
                        const int& sgcw,
                        T* stencil);

    static void poisson_offdiag(const int& i0,
                                const int& i1,
                                const T* dx,
                                const T& beta,
                                const int& sgcw,
                                T* stencil);

    static void v1diag(const int& i0,
                       const int& i1,
                       const int& ai0,
                       const int& ai1,
                       const T& alpha,
                       const T* a,
                       const int& sgcw,
                       T* stencil);

    static void
    v2diag(const int& i0, const int& i1, const T& alpha, const int& sgcw, T* stencil);

    static void poisson_diag(const int& i0, const int& i1, const int& sgcw, T* stencil);

    static void adj_diag(const int& i0,
                         const int& i1,
                         const int& pi0,
                         const int& pi1,
                         const int& side,
                         const int& btype,
                         const int& exOrder,
                         const T* dx,
                         const T& beta,
                         const T* b0,
                         const int& sgcw,
                         T* stencil);

    static void adj_poisson_diag(const int& i0,
                                 const int& i1,
                                 const int& pi0,
                                 const int& pi1,
                                 const int& side,
                                 const int& btype,
                                 const int& exOrder,
                                 const T* dx,
                                 const T& beta,
                                 const int& sgcw,
                                 T* stencil);

    static void adj_cf_diag(const int& i0,
                            const int& i1,
                            const int& pi0,
                            const int& pi1,
                            const int& r,
                            const int& side,
                            const int& intOrder,
                            const T* dx,
                            const T& beta,
                            const T* b0,
                            const int& sgcw,
                            T* stencil);

    static void adj_poisson_cf_diag(const int& i0,
                                    const int& i1,
                                    const int& pi0,
                                    const int& pi1,
                                    const int& r,
                                    const int& side,
                                    const int& intOrder,
                                    const T* dx,
                                    const T& beta,
                                    const int& sgcw,
                                    T* stencil);

    static void adj_offdiag(const int& i0,
                            const int& i1,
                            const int& pi0,
                            const int& pi1,
                            const int& side,
                            const int& btype,
                            const int& exOrder,
                            const T* dx,
                            const T& dir_factor,
                            const T& neu_factor,
                            const T& beta,
                            const T* b0,
                            const int& sgcw,
                            T* stencil);

    static void adj_poisson_offdiag(const int& i0,
                                    const int& i1,
                                    const int& pi0,
                                    const int& pi1,
                                    const int& side,
                                    const int& btype,
                                    const int& exOrder,
                                    const T* dx,
                                    const T& dir_factor,
                                    const T& neu_factor,
                                    const T& beta,
                                    const int& sgcw,
                                    T* stencil);

    static void adj_cf_offdiag(const int& i0,
                               const int& i1,
                               const int& ci0,
                               const int& ci1,
                               const int& r,
                               const int& side,
                               const int& intOrder,
                               const int& sgcw,
                               T* stencil);

    static void readj_offdiag(const int& i0,
                              const int& i1,
                              const int& pi0,
                              const int& pi1,
                              const int& side,
                              const int& sgcw,
                              T* stencil);

    static void adj_cf_bdryrhs(const int& i0,
                               const int& i1,
                               const int& pi0,
                               const int& pi1,
                               const int& side,
                               const int& sgcw,
                               const T* stencil,
                               const int& gcw,
                               const T* u,
                               T* rhs);
};
