#pragma once

template <typename T = double>
struct cd_stencil_2d_cuda {
    static void offdiag(const int& i0,
                        const int& j0,
                        const int& i1,
                        const int& j1,
                        const int& bi0,
                        const int& bj0,
                        const int& bi1,
                        const int& bj1,
                        const T* dx,
                        const T& beta,
                        const T* b0,
                        const T* b1,
                        const int& sgcw,
                        T* stencil);

    static void poisson_offdiag(const int& i0,
                                const int& j0,
                                const int& i1,
                                const int& j1,
                                const T* dx,
                                const T& beta,
                                const int& sgcw,
                                T* stencil);

    static void v1diag(const int& i0,
                       const int& j0,
                       const int& i1,
                       const int& j1,
                       const int& ai0,
                       const int& aj0,
                       const int& ai1,
                       const int& aj1,
                       const T& alpha,
                       const T* a,
                       const int& sgcw,
                       T* stencil);

    static void v2diag(const int& i0,
                       const int& j0,
                       const int& i1,
                       const int& j1,
                       const T& alpha,
                       const int& sgcw,
                       T* stencil);

    static void poisson_diag(const int& i0,
                             const int& j0,
                             const int& i1,
                             const int& j1,
                             const int& sgcw,
                             T* stencil);

    static void adj_diag(const int& i0,
                         const int& j0,
                         const int& i1,
                         const int& j1,
                         const int& pi0,
                         const int& pj0,
                         const int& pi1,
                         const int& pj1,
                         const int& dir,
                         const int& side,
                         const int& btype,
                         const int& exOrder,
                         const T* dx,
                         const T& beta,
                         const T* b0,
                         const T* b1,
                         const int& sgcw,
                         T* stencil);

    static void adj_poisson_diag(const int& i0,
                                 const int& j0,
                                 const int& i1,
                                 const int& j1,
                                 const int& pi0,
                                 const int& pj0,
                                 const int& pi1,
                                 const int& pj1,
                                 const int& dir,
                                 const int& side,
                                 const int& btype,
                                 const int& exOrder,
                                 const T* dx,
                                 const T& beta,
                                 const int& sgcw,
                                 T* stencil);

    static void adj_cf_diag(const int& i0,
                            const int& j0,
                            const int& i1,
                            const int& j1,
                            const int& pi0,
                            const int& pj0,
                            const int& pi1,
                            const int& pj1,
                            const int& r,
                            const int& dir,
                            const int& side,
                            const int& intOrder,
                            const T* dx,
                            const T& beta,
                            const T* b0,
                            const T* b1,
                            const int& sgcw,
                            T* stencil);

    static void adj_poisson_cf_diag(const int& i0,
                                    const int& j0,
                                    const int& i1,
                                    const int& j1,
                                    const int& pi0,
                                    const int& pj0,
                                    const int& pi1,
                                    const int& pj1,
                                    const int& r,
                                    const int& dir,
                                    const int& side,
                                    const int& intOrder,
                                    const T* dx,
                                    const T& beta,
                                    const int& sgcw,
                                    T* stencil);

    static void adj_offdiag(const int& i0,
                            const int& j0,
                            const int& i1,
                            const int& j1,
                            const int& pi0,
                            const int& pj0,
                            const int& pi1,
                            const int& pj1,
                            const int& dir,
                            const int& side,
                            const int& btype,
                            const int& exOrder,
                            const T* dx,
                            const T& dir_factor,
                            const T& neu_factor,
                            const T& beta,
                            const T* b0,
                            const T* b1,
                            const int& sgcw,
                            T* stencil);

    static void adj_poisson_offdiag(const int& i0,
                                    const int& j0,
                                    const int& i1,
                                    const int& j1,
                                    const int& pi0,
                                    const int& pj0,
                                    const int& pi1,
                                    const int& pj1,
                                    const int& dir,
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
                               const int& j0,
                               const int& i1,
                               const int& j1,
                               const int& ci0,
                               const int& cj0,
                               const int& ci1,
                               const int& cj1,
                               const int& r,
                               const int& dir,
                               const int& side,
                               const int& intOrder,
                               const int& sgcw,
                               T* stencil);

    static void readj_offdiag(const int& i0,
                              const int& j0,
                              const int& i1,
                              const int& j1,
                              const int& pi0,
                              const int& pj0,
                              const int& pi1,
                              const int& pj1,
                              const int& dir,
                              const int& side,
                              const int& sgcw,
                              T* stencil);

    static void adj_cf_bdryrhs(const int& i0,
                               const int& j0,
                               const int& i1,
                               const int& j1,
                               const int& pi0,
                               const int& pj0,
                               const int& pi1,
                               const int& pj1,
                               const int& dir,
                               const int& side,
                               const int& sgcw,
                               const T* stencil,
                               const int& gcw,
                               const T* u,
                               T* rhs);
};