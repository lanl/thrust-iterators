#pragma once

#include "submatrix_iterator.hpp"
#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

// Iterators over a transpose view of N dimensions.  This results in a random access,
// non-contiguous iterator
template <typename Iter, auto... I>
struct transpose_iterator
    : thrust::iterator_facade<transpose_iterator<Iter, I...>,
                              typename thrust::iterator_value<Iter>::type,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              typename thrust::iterator_reference<Iter>::type> {
    using diff_t = thrust::iterator_difference_t<Iter>;
    static constexpr auto N = sizeof...(I);

public:
    __host__ __device__ transpose_iterator(Iter first,
                                           const int (&sz)[N],
                                           const int (&st)[N])
        : first{first}, stride{st[I]...}, current{(0 * I)...}, n{sz[I]...}
    {
    }

    // __host__ __device__ void print_current() const
    // {
    //     printf(">>>>\ncurrent\t %d", current[0]);
    //     for (int i = 1; i < N; i++) { printf(" %d", current[i]); }
    //     printf("\n");
    //     printf("stride\t %d", stride[0]);
    //     for (int i = 1; i < N - 1; i++) { printf(" %d", stride[i]); }
    //     printf("\n");
    //     printf("n\t %d", n[0]);
    //     for (int i = 1; i < N; i++) { printf(" %d", n[i]); }
    //     printf("\n<<<<\n");
    // }

private:
    friend class thrust::iterator_core_access;
    template <typename, auto...>
    friend class transpose_iterator;

    __host__ __device__ typename thrust::iterator_reference<Iter>::type
    dereference() const
    {
        diff_t o = 0;
        for (int j = 0; j < N; j++) o += stride[j] * current[j];
        return *(first + o);
    }

    template <typename Other>
    __host__ __device__ bool equal(const transpose_iterator<Other, I...>& other) const
    {
        bool eq = true;
        for (int i = 0; i < N; i++) eq = eq && (current[i] == other.current[i]);
        return eq;
    }

    __host__ __device__ void increment()
    {
        // trying to avoid division and branching here
        // printf("incrementing..\n");
        // print_current();

        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(n[i] - 1 - current[i]); // will be 1 if this dimension is full
            int b = 1 - a * n[i];             // maps to 1 or -(n-1)
            int shift = f * b;
            // printf("a\t'%d'\tb\t'%d'\tshift\t'%d'\tf\t'%d'\n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by 1 then were done
            f *= !!(shift - 1);
        }
        // printf("before 'last' check f=%d\n", f);
        // print_current();
        //  if f is still 1 here then have incremented to the "last" position
        for (int i = 0; i < N - 1; i++) current[i] += f * (n[i] - 1);
        current[N - 1] += f * n[N - 1];
    }

    __host__ __device__ void decrement()
    {
        // printf("decrementing..\n");
        // print_current();
        //  trying to avoid division and branching here
        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(current[i]); // will be 1 if this dimension is empty
            int b = -1 + a * n[i]; // maps to -1 or +(n-1)
            int shift = f * b;
            // printf("a\t'%d'\tb\t'%d'\tshift\t'%d'\tf\t'%d'\n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by -1 then were done
            f *= !!(shift + 1);
        }
    }

    __host__ __device__ void advance(diff_t dist)
    {
        dist += detail::ravel<N>(n, current);
        // printf("distance %ld\n", dist);
        detail::unravel<N>(n, dist, current);
    }

    template <typename Other>
    __host__ __device__ diff_t
    distance_to(const transpose_iterator<Other, I...>& other) const
    {
        // assumes that first, n, and stride are identical between the two.
        return detail::ravel<N>(n, other.current) - detail::ravel<N>(n, current);
    }

    Iter first;
    int stride[N];
    int current[N]; // current index relative to base
    int n[N];       // transpose size
};

namespace detail
{
template <auto... I>
struct make_transpose_fn {
    static constexpr auto N = sizeof...(I);

    template <typename Iter>
    transpose_iterator<Iter, I...> operator()(Iter it, const int (&sz)[N]) const
    {
        int stride[N];
        stride[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--) { stride[i] = stride[i + 1] * sz[i + 1]; }

        return {it, sz, stride};
    }
};

} // namespace detail

template <auto... I>
static constexpr auto make_transpose = detail::make_transpose_fn<I...>{};
