#pragma once

template <typename T = double>
struct cd_stencil_coeffs_1d_cuda{
    static void offdiag1d(const int&,
                          const int&,
                          const int&,
                          const int&,
                          const T*,
                          const T&,
                          const T*,
                          const int&,
                          T*);
};
