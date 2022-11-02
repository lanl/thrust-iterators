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

#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

#include <utility>

template <auto N>
__host__ __device__ void stride_from_size(const int (&size)[N], int (&stride)[N])
{
    stride[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) stride[i] = stride[i + 1] * size[i + 1];
}

// compute the stride associated with dimension I
template <auto I, auto N, typename T>
__host__ __device__ T stride_dim(const T (&sz)[N])
{
    T t{1};
    for (int i = I + 1; i < N; i++) t *= sz[i];
    return t;
}

template <auto I, auto N, typename T>
__host__ __device__ T stride_dim(const T (&sz)[N], const T(&stm)[N])
{
    T t{1};
    for (int i = I + 1; i < N; i++) t *= sz[i];
    return t * stm[I];
}


template <auto N, typename T>
__host__ __device__ T stride_dim(const T (&sz)[N], int I)
{
    T t{1};
    for (int i = I + 1; i < N; i++) t *= sz[i];
    return t;
}


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
