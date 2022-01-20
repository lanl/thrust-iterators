#pragma once

template <typename T = double>
struct sliding_test_cuda {
    static void init(T*, int, int);

    static void transform(const T*, int, T*, int, int);
};
