#pragma once

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

namespace detail
{

// general utility for "raveling" an N-D coordinate into a single index assuming zero
// based offset
template <auto N>
__host__ __device__ int ravel(const int (&size)[N], const int (&coord)[N])
{
    int t{0};
    for (int i = 0; i < N - 1; i++) t = (t + coord[i]) * size[i + 1];
    return t + coord[N - 1];
}

// utility for "unraveling" an index into an N-D coordinate
template <auto N>
__host__ __device__ void unravel(const int (&size)[N], int index, int (&coord)[N])
{
    if constexpr (N == 1) {
        coord[0] = index;
    } else if constexpr (N == 2) {
        // presumably division and modulus are a single operation
        coord[0] = index / size[1];
        coord[1] = index % size[1];
    } else if constexpr (N == 3) {
        coord[0] = index / (size[1] * size[2]);
        index = index % (size[1] * size[2]);
        coord[1] = index / size[2];
        coord[2] = index % size[2];
    } else {
        for (int i = 0; i < N - 1; i++) {
            int sz = size[i + 1];
            for (int j = i + 2; j < N; j++) sz *= size[j];
            coord[i] = index / sz;
            index = index % sz;
        }
        coord[N - 1] = index;
    }

    if constexpr (N > 1) {
        // adjust for unraveling the "last" position
        int f = !(size[0] - coord[0]);
        coord[0] -= f;
        for (int i = 1; i < N - 1; i++) coord[i] += f * (size[i] - 1);
        coord[N - 1] += f * size[N - 1];
    }
}

} // namespace detail

// Iterators over a submatrix of N dimensions.  This results in a random access,
// non-contiguous iterator
template <typename Iter, int N>
struct submatrix_iterator
    : thrust::iterator_facade<submatrix_iterator<Iter, N>,
                              typename thrust::iterator_value<Iter>::type,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              typename thrust::iterator_reference<Iter>::type> {
    using diff_t = thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__ submatrix_iterator(Iter first,
                                           const int (&sz)[N],
                                           const int (&lb)[N],
                                           const int (&ub)[N])
        : first{first + detail::ravel<N>(sz, lb)}
    {

        for (int i = 0; i < N; i++) {
            n[i] = ub[i] - lb[i] + 1;
            current[i] = 0;
        }
        stride[N - 2] = sz[N - 1];

        for (int i = N - 3; i >= 0; i--) { stride[i] = stride[i + 1] * sz[i + 1]; }
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
    template <typename, int>
    friend class submatrix_iterator;

    __host__ __device__ typename thrust::iterator_reference<Iter>::type
    dereference() const
    {
        diff_t o = current[N - 1];
        for (int j = 0; j < N - 1; j++) o += stride[j] * current[j];
        return *(first + o);
    }

    template <typename Other>
    __host__ __device__ bool equal(const submatrix_iterator<Other, N>& other) const
    {
        bool eq = true;
        for (int i = 0; i < N; i++) eq = eq && (current[i] == other.current[i]);
        return eq;
    }

    __host__ __device__ void increment()
    {
        // trying to avoid division and branching here
        //printf("incrementing..\n");
        //print_current();

        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(n[i] - 1 - current[i]); // will be 1 if this dimension is full
            int b = 1 - a * n[i];             // maps to 1 or -(n-1)
            int shift = f * b;
            //printf("a\t'%d'\tb\t'%d'\tshift\t'%d'\tf\t'%d'\n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by 1 then were done
            f *= !!(shift - 1);
        }
        //printf("before 'last' check f=%d\n", f);
        //print_current();
        // if f is still 1 here then have incremented to the "last" position
        for (int i = 0; i < N - 1; i++) current[i] += f * (n[i] - 1);
        current[N - 1] += f * n[N - 1];
    }

    __host__ __device__ void decrement()
    {
        //printf("decrementing..\n");
        //print_current();
        // trying to avoid division and branching here
        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(current[i]); // will be 1 if this dimension is empty
            int b = -1 + a * n[i]; // maps to -1 or +(n-1)
            int shift = f * b;
            //printf("a\t'%d'\tb\t'%d'\tshift\t'%d'\tf\t'%d'\n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by -1 then were done
            f *= !!(shift + 1);
        }
    }

    __host__ __device__ void advance(diff_t dist)
    {
        dist += detail::ravel<N>(n, current);
        //printf("distance %ld\n", dist);
        detail::unravel<N>(n, dist, current);
    }

    template <typename Other>
    __host__ __device__ diff_t
    distance_to(const submatrix_iterator<Other, N>& other) const
    {
        // assumes that first, n, and stride are identical between the two.
        return detail::ravel<N>(n, other.current) - detail::ravel<N>(n, current);
    }

    Iter first;
    int stride[N - 1];
    int current[N]; // current index relative to base
    int n[N];       // submatrix size
};

template <typename Iter, int N>
submatrix_iterator<Iter, N>
make_submatrix(Iter it, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N])
{
    return {it, sz, lb, ub};
}