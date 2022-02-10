#pragma once

#include <type_traits>

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
