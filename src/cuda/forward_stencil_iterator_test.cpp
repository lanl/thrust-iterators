#include "../forward_stencil_test.hpp"
#include "forward_stencil_iterator.hpp"
#include "matrix_traversal_iterator.hpp"

#include <thrust/device_vector.h>

template <typename T, auto N, int I>
void forward_stencil_test_cuda<T, N, I>::init(const T* v, const int (&dims)[N])
{
    auto n = stride_dim<-1, N>(dims);
    auto limit = stride_dim<I - 1, N>(dims) - 1;
    auto stride = stride_dim<I, N>(dims);
    // compute size of stencil output
    auto u_sz = 1;
    for (int i = 0; i < N; i++) { u_sz *= i == I ? dims[i] - 1 : dims[i]; }

    thrust::device_vector<T> u(v, v + n);
    auto s = make_forward_stencil(u.begin(), stride, limit, u_sz);
    auto a = s + 1;
    assert(thrust::distance(s, a) == 1);
    --a;
    assert(a == s);
    ++a;
    --a;
    assert(a == s);
    assert(thrust::distance(s, a) == 0);

    for (int i = 0; i < u_sz; i++) assert(thrust::distance(s, s + i) == i);
    auto ss = s + (u_sz - 1);
    assert(thrust::distance(s, ss) == u_sz - 1);
    ++ss;

    assert(thrust::distance(s, ss) == u_sz);
    assert(thrust::distance(s, s + u_sz) == u_sz);

    // go to the edge to test rollover
    if constexpr (I != 0) {
        a = s + limit - 1;
        assert(thrust::distance(s, a) == limit - 1);
        ++a;
        --a;
        assert(thrust::distance(s, a) == limit - 1);
        auto b = a + 1;
        --b;
        assert(b == a);

        b = s + limit;
        b = b - 1;
        assert(thrust::distance(s, b) == limit - 1);
        assert(thrust::distance(b, s) == 1 - limit);
        ++b;
        assert(thrust::distance(s, b) == limit);
        assert(thrust::distance(b, s) == -limit);
        ++b;
        assert(thrust::distance(s, b) == limit + 1);
        assert(thrust::distance(b, s) == -limit - 1);
    }
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

template <typename T, auto N, int I>
void forward_stencil_test_cuda<T, N, I>::transform(const T* v, const int (&dims)[N], T* u)
{
    auto n = stride_dim<-1, N>(dims);
    int udims[N];
    for (int i = 0; i < N; i++) udims[i] = i == I ? dims[i] - 1 : dims[i];

    auto limit = stride_dim<I - 1, N>(udims);
    auto stride = stride_dim<I, N>(dims);
    auto u_sz = stride_dim<-1, N>(udims);

    thrust::device_vector<T> x(v, v + n);
    thrust::device_vector<T> y(u, u + u_sz);

    auto s = make_forward_stencil(x.begin(), stride, limit, u_sz);

    assert(thrust::distance(s, s + u_sz) == u_sz);
    thrust::transform(s, s + u_sz, y.begin(), gg<T>{});
    thrust::copy(y.begin(), y.end(), u);

    auto z = 3 + (*s) * 4.0;
}

template struct forward_stencil_test_cuda<double, 1, 0>;
template struct forward_stencil_test_cuda<double, 2, 0>;
template struct forward_stencil_test_cuda<double, 2, 1>;
template struct forward_stencil_test_cuda<double, 3, 0>;
template struct forward_stencil_test_cuda<double, 3, 1>;
template struct forward_stencil_test_cuda<double, 3, 2>;
