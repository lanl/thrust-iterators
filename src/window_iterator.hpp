#pragma once

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

// divides an iterator into a number of contigous, non-overlapping windows
// assumes the window size evenly divides the iterator size
template <typename Iter>
struct window_iterator
    : thrust::iterator_facade<window_iterator<Iter>,
                              Iter,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              Iter> {
    using diff_t = thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__ window_iterator(Iter first, diff_t window_size)
        : first{first}, w{window_size}
    {
    }

private:
    Iter first;
    diff_t w;

    friend class thrust::iterator_core_access;
    template <typename>
    friend class window_iterator;

    __host__ __device__ Iter dereference() const { return first; }

    template <typename Other>
    __host__ __device__ bool equal(const window_iterator<Other>& other) const
    {
        return this->first == other.first;
    }

    __host__ __device__ void increment() { thrust::advance(first, w); }

    __host__ __device__ void decrement() { thrust::advance(first, -w); }

    __host__ __device__ void advance(diff_t n) { thrust::advance(first, n * w); }

    template <typename Other>
    __host__ __device__ diff_t distance_to(const window_iterator<Other>& other) const
    {
        return thrust::distance(first, other.first) / w;
    }
};

template <typename Iter>
window_iterator<Iter> make_window(Iter it, int window_size)
{
    return window_iterator<Iter>(it, window_size);
}

template <typename Iter>
struct window_pair {
    window_iterator<Iter> first;
    window_iterator<Iter> last;
};

template <typename C>
auto make_window_pair(C&& c, int window_size) -> window_pair<decltype(c.begin())>
{
    return {make_window(c.begin(), window_size), make_window(c.end(), window_size)};
}
