#pragma once

template <typename T, auto N, int I>
struct forward_stencil_test_cuda {
    static void init(const T*, const int (&)[N]);

    static void transform(const T*, const int (&)[N], T*);
};
