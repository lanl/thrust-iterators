
#include "window_iterator.hpp"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include "window_iterator_test.hpp"

template <typename T>
void window_test_cuda<T>::init(T* v, int n)
{
    thrust::device_vector<T> u(v, v + n);

    auto window = make_window(u.begin(), n / 2);
    auto it = *window;
    *it = 1.0;
    it[1] = 2.0;

    it = *++window;
    *it = 3.0;
    it[1] = 4.0;

    thrust::copy(u.begin(), u.end(), v);
}

template <typename T>
struct ff {
    T x, y;

    ff(T x, T y) : x{x}, y{y} {}

    template <typename It>
    __host__ __device__ void operator()(It it)
    {
        it[0] = x;
        it[1] = y;
    }
};

template <typename T>
void window_test_cuda<T>::transform(T* v, int n)
{
    thrust::device_vector<T> u(v, v + n);

    int window_size = 2;
    thrust::for_each(make_window(u.begin(), window_size),
                     make_window(u.end(), window_size),
                     ff(1.0, 2.0));
    thrust::copy(u.begin(), u.end(), v);
}

template struct window_test_cuda<double>;
