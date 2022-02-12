#pragma once

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

template <typename T>
struct stencil_t {
    T a, b;
};

namespace detail
{
template <typename Iter>
using iter_val = typename thrust::iterator_value_t<Iter>;

template <typename Iter>
using iter_ref = typename thrust::iterator_reference_t<Iter>;

template <typename Iter>
using stencil_val = stencil_t<iter_val<Iter>>;

template <typename Iter>
using stencil_ref = stencil_t<iter_ref<Iter>>;

} // namespace detail

// Forward stencil iterator returns a tuple of references to the current item and
// the offset item.  This could be generalized to more items but we only need 2 for now
template <typename Iter>
struct forward_stencil_iterator
    : thrust::iterator_facade<forward_stencil_iterator<Iter>,
                              detail::stencil_val<Iter>,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              detail::stencil_ref<Iter>> {
    using diff_t = thrust::iterator_difference_t<Iter>;
    using ref_t = detail::stencil_ref<Iter>;
    using val_t = detail::stencil_val<Iter>;

public:
    __host__ __device__ forward_stencil_iterator(Iter first,
                                                 diff_t stencil_stride,
                                                 diff_t max_stride_distance)
        : first{first},
          stencil_stride{stencil_stride},
          max_stride_distance{max_stride_distance},
          current_distance{0}
    {
    }

    __host__ __device__ void print_current() const
    {
        printf("\n  current stencil: %d / %d / %d\n",
               stencil_stride,
               max_stride_distance,
               current_distance);
    }

private:
    friend class thrust::iterator_core_access;
    template <typename>
    friend class forward_stencil_iterator;

    __host__ __device__ ref_t dereference() const
    {
        return {*first, *(first + stencil_stride)};
    }

    template <typename Other>
    __host__ __device__ bool equal(const forward_stencil_iterator<Other>& other) const
    {
        return first == other.first;
    }

    __host__ __device__ void increment()
    {
        ++first;
        ++current_distance;
        // If we've reached max_stride_distance then we need to increment by stride and
        // reset our distance
        int distance_reached = !(max_stride_distance - current_distance);
        current_distance -= distance_reached * max_stride_distance;
        first += distance_reached * stencil_stride;
    }

    __host__ __device__ void decrement()
    {
        --first;
        --current_distance;
        // If current_distance == -1, then we need to decrement by stride and reset the
        // distance to skip over the "bad" point
        int distance_reached = !(1 + current_distance);
        current_distance += distance_reached * max_stride_distance;
        first -= distance_reached * stencil_stride;
    }

    __host__ __device__ void advance(diff_t dist)
    {
        // printf("\nadvancing dist %d", dist);
        // print_current();
        // need to add correct multiples of the stencil_stride to compensate for dist
        // causing our stencil to "rollover".
        auto base_dist = dist + current_distance;
        int backwards = base_dist < 0;
        auto iter_dist = dist + (base_dist / max_stride_distance) * stencil_stride -
                         backwards * stencil_stride;
        current_distance =
            backwards * max_stride_distance + base_dist % max_stride_distance;

        thrust::advance(first, iter_dist);
    }

    template <typename Other>
    __host__ __device__ diff_t
    distance_to(const forward_stencil_iterator<Other>& other) const
    {
        auto iter_dist = thrust::distance(first, other.first);

        auto n = max_stride_distance - 1;
        auto m =
            (iter_dist - n + current_distance - stencil_stride - other.current_distance) /
            (stencil_stride + n);
        auto actual_dist = iter_dist - (m + 1) * stencil_stride;

        return actual_dist;
    }

    Iter first;
    diff_t stencil_stride;
    diff_t max_stride_distance;
    diff_t current_distance;
};

template <typename Iter>
forward_stencil_iterator<Iter>
make_forward_stencil(Iter it,
                     typename thrust::iterator_difference_t<Iter> stencil_stride,
                     typename thrust::iterator_difference_t<Iter> max_stride_distance)
{
    return {it, stencil_stride, max_stride_distance};
}
