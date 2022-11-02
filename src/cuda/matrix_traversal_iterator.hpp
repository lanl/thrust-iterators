// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#pragma once

#include <utility>

#include <thrust/advance.h>
#include <thrust/distance.h>

#include "forward_stencil_iterator.hpp"
#include "transpose_iterator.hpp"

namespace detail
{

template <typename T, typename U, auto... I>
__host__ __device__ bool equal_(std::index_sequence<I...>, T&& x, U&& y)
{
    {
        return ((x[I] == y[I]) && ...);
    }
}

template <typename T, auto N>
__host__ __device__ bool equal(const T (&x)[N], const T (&y)[N])
{
    return equal_(std::make_index_sequence<N>{}, x, y);
}

} // namespace detail

template <typename Iter, auto N, typename Val, typename Ref>
struct matrix_traversal_iterator
    : thrust::iterator_facade<matrix_traversal_iterator<Iter, N, Val, Ref>,
                              Val,
                              typename thrust::iterator_system<Iter>::type,
                              typename thrust::iterator_traversal<Iter>::type,
                              Ref> {
    using diff_t = thrust::iterator_difference_t<Iter>;

public:
    __host__ __device__ matrix_traversal_iterator(Iter first,
                                                  const int (&stride)[N],
                                                  const int (&start)[N],
                                                  const int (&n)[N])
        : first{first}
    {
        for (int i = 0; i < N; i++) {
            this->stride[i] = stride[i];
            this->current[i] = start[i];
            this->n[i] = n[i];
        }

        // printf("Creating a submatrix iterator with://n");
        // printf("//tStride: %i", stride[0]);
        // for (int i = 1; i < N; i++) printf(" %i", stride[i]);
        // printf("//n//tSize: %i", n[0]);
        // for (int i = 1; i < N; i++) printf(" %i", n[i]);
        // printf("//n");
    }

    __host__ __device__ int stride_dim(int i) const { return stride[i]; }

    __host__ __device__ int size() const { return ::stride_dim<-1, N>(n); }

    __host__ __device__ forward_stencil_iterator<matrix_traversal_iterator> stencil(int I)
    {
        int dims[N];
        for (int i = 0; i < N; i++) dims[i] = i == I ? n[i] - 1 : n[i];

        auto stride = ::stride_dim<N>(n, I);
        auto limit = ::stride_dim<N>(dims, I - 1);
        auto sz = ::stride_dim<N>(dims, -1);

        return make_forward_stencil(*this, stride, limit, sz);
    }

    template <auto I>
    __host__ __device__ forward_stencil_iterator<matrix_traversal_iterator> stencil()
    {
        int dims[N];
        for (int i = 0; i < N; i++) dims[i] = i == I ? n[i] - 1 : n[i];

        auto stride = ::stride_dim<I, N>(n);
        auto limit = ::stride_dim<I - 1, N>(dims);
        auto sz = ::stride_dim<-1, N>(dims);

        return make_forward_stencil(*this, stride, limit, sz);
    }

private:
    friend class thrust::iterator_core_access;
    template <typename, auto, typename, typename>
    friend class matrix_traversal_iterator;

    __host__ __device__ Ref // typename thrust::iterator_reference<Iter>::type
    dereference() const
    {
        diff_t o = 0;
        for (int j = 0; j < N; j++) o += stride[j] * current[j];

        if constexpr (std::is_same_v<Ref, typename thrust::iterator_reference_t<Iter>>)
            return *(first + o);
        else
            return first + o;
    }

    template <typename Other>
    __host__ __device__ bool equal(const matrix_traversal_iterator<Other, N>& other) const
    {
        return detail::equal(current, other.current);
    }

    __host__ __device__ void increment()
    {
        // trying to avoid division and branching here
        // printf("incrementing..//n");
        // print_current();

        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(n[i] - 1 - current[i]); // will be 1 if this dimension is full
            int b = 1 - a * n[i];             // maps to 1 or -(n-1)
            int shift = f * b;
            // printf("a//t'%d'//tb//t'%d'//tshift//t'%d'//tf//t'%d'//n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by 1 then were done
            f *= !!(shift - 1);
        }
        // printf("before 'last' check f=%d//n", f);
        // print_current();
        //  if f is still 1 here then have incremented to the "last" position
        for (int i = 0; i < N - 1; i++) current[i] += f * (n[i] - 1);
        current[N - 1] += f * n[N - 1];
    }

    __host__ __device__ void decrement()
    {
        // printf("decrementing..//n");
        // print_current();
        //  trying to avoid division and branching here
        int f = 1;
        for (int i = N - 1; i >= 0; i--) {
            int a = !(current[i]); // will be 1 if this dimension is empty
            int b = -1 + a * n[i]; // maps to -1 or +(n-1)
            int shift = f * b;
            // printf("a//t'%d'//tb//t'%d'//tshift//t'%d'//tf//t'%d'//n", a, b, shift, f);
            current[i] += shift;
            // if we shifted by -1 then were done
            f *= !!(shift + 1);
        }
    }

    __host__ __device__ void advance(diff_t dist)
    {
        dist += ravel<N>(n, current);
        // printf("distance %ld//n", dist);
        unravel<N>(n, dist, current);
    }

    template <typename Other>
    __host__ __device__ diff_t
    distance_to(const matrix_traversal_iterator<Other, N>& other) const
    {
        // assumes that first, n, and stride are identical between the two.
        return ravel<N>(n, other.current) - ravel<N>(n, current);
    }

    Iter first;
    int stride[N];
    int current[N]; // current index relative to base
    int n[N];       // transpose size
};
