#pragma once

template <typename T, auto... I>
struct transpose_test_cuda {
    static constexpr auto N = sizeof...(I);

    static void init(const T*, const int (&sz)[N], T*);

    //static void tabulate(T*, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N]);
};
