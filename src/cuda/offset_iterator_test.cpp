#include "../offset_iterator_test.hpp"
#include "matrix_utils.hpp"
#include "md_device_span.hpp"
#include "offset_iterator.hpp"
#include "matrix_traversal_iterator.hpp"

// size of offset is n_off * N
template <typename T, auto N>
void offset_test_cuda<T, N>::init(T* v, int n_off, int* offset, const int (&dims)[N])
{
    int stride[N];
    stride_from_size(dims, stride);

    // printf("offsets://n %d", *offset);
    // for (int i = 1; i < n_off; i++)
    //     printf(" %d", *(offset + i));
    // printf("//n");

    auto o = make_offset_span(offset, n_off, N);
    auto u = thrust::device_ptr<T>(v);
    auto it = make_offset_iterator(u, stride, o.begin());

    assert(thrust::distance(it, it + 1) == 1);

    {
        auto x = it;
        ++x;
        --x;
        assert(x == it);
        assert(thrust::distance(x, it) == 0);
    }

    for (int i = 0; i < n_off; i++) { it[i] = -1 - i; }
}


template struct offset_test_cuda<double, 1>;
template struct offset_test_cuda<double, 2>;
template struct offset_test_cuda<double, 3>;
