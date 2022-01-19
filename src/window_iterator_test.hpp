#pragma once

template <typename T = double>
struct window_test_cuda {
    static void init(T*, int);

    static void transform(T*, int);
};
