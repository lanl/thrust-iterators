#pragma once

#include "window_iterator.hpp"
#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

// a sliding window returns an iterator of window size upon dereferencing.  The sliding
// windows overlap with eachother
template <typename Iter>
struct sliding_iterator
    : thrust::iterator_facade<sliding_iterator<Iter>,
                              Iter,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              Iter> {
    using diff_t = thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__ sliding_iterator(Iter first, diff_t window_size)
        : first{first}, w{window_size}
    {
    }

private:
    Iter first;
    diff_t w;

    friend class thrust::iterator_core_access;
    template <typename>
    friend class sliding_iterator;

    __host__ __device__ Iter dereference() const { return first; }

    template <typename Other>
    __host__ __device__ bool equal(const sliding_iterator<Other>& other) const
    {
        return this->first == other.first;
    }

    __host__ __device__ void increment() { thrust::advance(first, 1); }

    __host__ __device__ void decrement() { thrust::advance(first, -1); }

    __host__ __device__ void advance(diff_t n) { thrust::advance(first, n); }

    template <typename Other>
    __host__ __device__ diff_t distance_to(const sliding_iterator<Other>& other) const
    {
        return thrust::distance(first, other.first);
    }
};

template <typename Iter>
sliding_iterator<Iter> make_sliding(Iter it, int window_size)
{
    return sliding_iterator<Iter>(it, window_size);
}

// Since we only allow full sliding window, the check first == last needs to actually be
// different from something like first == first.  We handle that here by adjusting where
// the container's last iterator.  If the user constructs their own 'last' iterator, they
// are in for a nasty surprise...

template <typename Iter>
struct sliding_pair {
    sliding_iterator<Iter> first;
    sliding_iterator<Iter> last;
};

template <typename C>
auto make_sliding_pair(C&& c, int window_size) -> sliding_pair<decltype(c.begin())>
{
    return {make_sliding(c.begin(), window_size),
            make_sliding(c.end() - window_size + 1, window_size)};
}
