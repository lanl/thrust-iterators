#include "sliding_iterator_test.hpp"
#include "sliding_iterator.hpp"

#include <thrust/device_vector.h>

template<typename T>
void sliding_test_cuda<T>::init(T* v, int n, int window_size) {
    thrust::device_vector<T> u(v, v+ n);
    auto s = make_sliding(u.begin(), window_size);

}

template struct sliding_test_cuda<double>;
