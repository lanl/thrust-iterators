#pragma once

#include <utility>

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>

#include "matrix_utils.hpp"

namespace detail
{
__host__ __device__ int inline coarse_index(int r, int fi)
{
    // if (fi < 0)
    //     ci = (fi + 1) / r - 1;
    // else
    //     ci = fi / r;
    int negative_fi = (fi < 0);
    return (1 - negative_fi) * (fi / r) + negative_fi * ((fi + 1) / r - 1);
}

} // namespace detail

template <typename Iter>
struct coarse_to_fine_iterator
    : thrust::iterator_facade<coarse_to_fine_iterator<Iter>,
                              typename thrust::iterator_value_t<Iter>,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              typename thrust::iterator_reference_t<Iter>> {
    using diff_t = thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__
    coarse_to_fine_iterator(Iter first, int ratio, int stride, int fine_index)
        : first{first},
          r{ratio},
          s{stride},
          fi{fine_index},
          ci{detail::coarse_index(ratio, fine_index)}
    {
    }

private:
    friend class thrust::iterator_core_access;
    template <typename>
    friend class coarse_to_fine_iterator;

    __host__ __device__ typename thrust::iterator_reference_t<Iter> dereference() const
    {
        return *(first + (detail::coarse_index(r, fi) - ci) * s);
    }

    template <typename Other>
    __host__ __device__ bool equal(const coarse_to_fine_iterator<Other>& other) const
    {
        return fi == other.fi;
    }

    __host__ __device__ void increment() { ++fi; }

    __host__ __device__ void decrement() { --fi; }

    __host__ __device__ void advance(diff_t dist) { fi += dist; }

    template <typename Other>
    __host__ __device__ diff_t
    distance_to(const coarse_to_fine_iterator<Other>& other) const
    {
        // assumes that first, n, and stride are identical between the two.
        return other.fi - fi;
    }

    Iter first;
    int r;
    int s;
    int fi; // fine index
    int ci; // base coarse index
};

template <auto I, auto N, typename Iter>
coarse_to_fine_iterator<Iter>
make_coarse_to_fine(Iter it, const int (&sz)[N], const int (&ci)[N], int fi, int ratio)
{
    return coarse_to_fine_iterator<Iter>{
        it + ravel<N>(sz, ci), ratio, stride_dim<I, N>(sz), fi};
}
