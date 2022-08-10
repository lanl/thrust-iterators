#pragma once

template <typename T, auto N>
struct offset_test_cuda {
    static void init(T* v, int n_off, int* offset, const int (&dims)[N]);
};
