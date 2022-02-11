#include "../forward_stencil_test.hpp"
#include "forward_stencil_iterator.hpp"

#include <thrust/device_vector.h>

template <typename T>
void forward_stencil_test_cuda<T>::init(const T* v, int n)
{
    thrust::device_vector<T> u(v, v + n);
    auto s = make_forward_stencil(u.begin(), n / 2, n - 1);
    auto a = s + 1;
    assert(thrust::distance(s, a) == 1);
    ++s;
    assert(a == s);
    --a;
    ++a;
    assert(a == s);
    assert(thrust::distance(s, a) == 0);
}

template <typename T>
void forward_stencil_test_cuda<T>::init2D(const T* v, int dims[2])
{
    auto n = dims[1] * dims[0];
    auto limit = dims[1] - 1;

    thrust::device_vector<T> u(v, v + n);
    auto s = make_forward_stencil(u.begin(), 1, limit);
    auto a = s + 1;
    assert(thrust::distance(s, a) == 1);
    --a;
    assert(a == s);
    ++a;
    --a;
    assert(a == s);
    assert(thrust::distance(s, a) == 0);

    // go to the edge to test rollover
    printf("\nchecking limit\n");
    a.print_current();
    a = s + (limit - 1);
    a.print_current();
    assert(thrust::distance(s, a) == limit - 1);
    ++a;
    a.print_current();
    --a;
    assert(thrust::distance(s, a) == limit - 1);
    auto b = a + 1;
    --b;
    assert(b == a);

    b = s + limit;
    b.print_current();
    b = b - 1;
    b.print_current();

}

template <typename T>
struct gg {

    template <typename Tuple>
    __host__ __device__ T operator()(Tuple&& t) const
    {
        auto&& [a, b] = t;

        return b - a;
    }
};

template <typename T>
void forward_stencil_test_cuda<T>::transform(const T* v, int nv, int stride, T* u)
{
    thrust::device_vector<T> x(v, v + nv);
    thrust::device_vector<T> y(nv - stride);

    auto s = make_forward_stencil(x.begin(), stride, nv - 1);

    // assert(thrust::distance(s, s + nv - stride) == nv - stride);
    // thrust::transform(s, s + nv - stride, u, gg<T>{});
}

template struct forward_stencil_test_cuda<double>;
