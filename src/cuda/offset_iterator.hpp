#pragma once

#include <utility>

#include <thrust/advance.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_facade.h>

#include "traits.hpp"


template <typename Iter, auto N>
struct offset_iterator
    : thrust::iterator_facade<offset_iterator<Iter, N>,
                              typename thrust::iterator_value_t<Iter>,
                              typename thrust::iterator_system_t<Iter>,
                              typename thrust::iterator_traversal<Iter>::type,
                              typename thrust::iterator_reference_t<Iter>> {

    using ref_t = typename thrust::iterator_reference_t<Iter>;
    using diff_t = typename thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__ offset_iterator(Iter first,
                                        const int (&stride)[N],
                                        thrust::device_ptr<int> o)
        : first{first}, o{o}, o_idx{0}
    {
        for (int i = 0; i < N; i++) this->stride[i] = stride[i];
    }

private:
    friend class thrust::iterator_core_access;

    template <typename, auto>
    friend class offset_iterator;

    __host__ __device__ ref_t dereference() const
    {
        diff_t i = 0;
        for (int j = 0; j < N; j++) i += stride[j] * o[o_idx * N + j];

        return *(first + i);
    }

    template <typename Other>
    __host__ __device__ bool equal(const offset_iterator<Other, N>& other) const
    {
        return other.o_idx == o_idx;
    }

    __host__ __device__ void increment() { ++o_idx; }

    __host__ __device__ void decrement() { --o_idx; }

    __host__ __device__ void advance(diff_t dist) { o_idx += dist; }

    template <typename Other>
    __host__ __device__ diff_t distance_to(const offset_iterator<Other, N>& other) const
    {
        return other.o_idx - o_idx;
    }

    Iter first;
    int stride[N];
    thrust::device_ptr<int> o;
    int o_idx;
};

template <typename Iter, int N>
offset_iterator<Iter, N>
make_offset_iterator(Iter it, const int (&stride)[N], thrust::device_ptr<int> o)
{
    return {it, stride, o};
}
