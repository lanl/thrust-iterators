#pragma once

template <typename T>
struct test {
    static void init(int fi0, int fi1, int ci0, int ci1, int ratio, const T* coarse, T* fine);
};
