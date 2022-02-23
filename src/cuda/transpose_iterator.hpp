#pragma once

#include "matrix_utils.hpp"

template <typename, auto>
struct matrix_traversal_iterator;

namespace detail
{
template <auto... I>
struct make_transpose_fn {
    static constexpr auto N = sizeof...(I);

    template <typename Iter>
    __host__ __device__ matrix_traversal_iterator<Iter, N>
    operator()(Iter it, const int (&sz)[N]) const
    {
        int stride[] = {stride_dim<I, N>(sz)...};
        int current[] = {(0 * I)...};
        int n[] = {sz[I]...};
        return {it, stride, current, n};
    }
};

} // namespace detail

template <auto... I>
static __device__ constexpr auto make_transpose = detail::make_transpose_fn<I...>{};
