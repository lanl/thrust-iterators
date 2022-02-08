#include "sliding_iterator.hpp"
#include "../sliding_iterator_test.hpp"

#include <thrust/device_vector.h>

template <typename T>
void sliding_test_cuda<T>::init(T* v, int n, int window_size)
{
    thrust::device_vector<T> u(v, v + n);
    auto s = make_sliding(u.begin(), window_size);
}

template <typename T>
struct gg {
    int w;

    template <typename It>
    __host__ __device__ T operator()(It it) {
        T t{};
        for (int i = 0; i < w; i++)
            t += it[i];
        return t;
    }
};

template <typename T>
void sliding_test_cuda<T>::transform(const T* v, int nv, T* u, int nu, int window_size)
{
    thrust::device_vector<T> x(v, v + nv);
    thrust::device_vector<T> y(u, u + nu);
    auto && [first, last] = make_sliding_pair(x, window_size);
    thrust::transform(first, last, y.begin(), gg<T>{window_size});

    thrust::copy(y.begin(), y.end(), u);

}

template struct sliding_test_cuda<double>;
