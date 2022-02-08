#include "window_iterator.hpp"

#include <thrust/device_vector.h>

template <typename T>
void test_construction(T* v, int n) {
    thrust::device_vector(v, v + n);
}
