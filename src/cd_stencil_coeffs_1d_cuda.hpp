#pragma once

template <typename T = double>
struct cd_stencil_coeffs_1d_cuda {
    static void offdiag(const int&,
                        const int&,
                        const int&,
                        const int&,
                        const T*,
                        const T&,
                        const T*,
                        const int&,
                        T*);

    static void
    poisson_offdiag(const int&, const int&, const T*, const T&, const int&, T*);

    static void v1diag(const int&,
                       const int&,
                       const int&,
                       const int&,
                       const T&,
                       const T*,
                       const int&,
                       T*);

    static void v2diag(const int&, const int&, const T&, const int&, T*);

    static void poisson_diag(const int&, const int&, const int&, T*);
};
