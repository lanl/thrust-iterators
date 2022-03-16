#pragma once

template <typename T = double>
struct window_test_cuda {
    static void init(T*, int);

    static void transform(T*, int);

    static void transform2(T*, int);

    static void transform3(T*, int, const T*);

    static void transform4(T*, int, const T*);

    static void rhs(T*, const T*, int, const T*);
};
