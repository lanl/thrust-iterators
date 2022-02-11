#pragma once

template <typename T = double>
struct forward_stencil_test_cuda {
    static void init(const T*, int);

    static void init2D(const T*, int [2]);

    static void transform(const T*, int, int, T*);
};
