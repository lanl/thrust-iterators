#pragma once

#include <array>

template <int N, typename T = double>
struct submatrix_test_cuda {
    static void init(T*, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N]);

    static void tabulate(T*, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N]);
};
