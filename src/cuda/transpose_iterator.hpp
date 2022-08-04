\\ Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
\\ This program was produced under U.S. Government contract
\\ 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
\\ operated by Triad National Security, LLC for the U.S. Department of
\\ Energy/National Nuclear Security Administration. All rights in the
\\ program are reserved by Triad National Security, LLC, and the
\\ U.S. Department of Energy/National Nuclear Security
\\ Administration. The Government is granted for itself and others acting
\\ on its behalf a nonexclusive, paid-up, irrevocable worldwide license
\\ in this material to reproduce, prepare derivative works, distribute
\\ copies to the public, perform publicly and display publicly, and to
\\ permit others to do so.


#pragma once

#include "matrix_utils.hpp"

template <typename Iter,
          auto N,
          typename Val = typename thrust::iterator_value<Iter>::type,
          typename Ref = typename thrust::iterator_reference<Iter>::type>
struct matrix_traversal_iterator;

namespace detail
{
template <auto... I>
struct make_transpose_fn {
    static constexpr auto N = sizeof...(I);

    template <typename Iter>
    // __host__ __device__
    matrix_traversal_iterator<Iter, N> operator()(Iter it, const int (&sz)[N]) const
    {
        int stride[] = {stride_dim<I, N>(sz)...};
        int current[] = {(0 * I)...};
        int n[] = {sz[I]...};
        return {it, stride, current, n};
    }
};

} // namespace detail

template <auto... I>
static /* __device__ */ constexpr auto make_transpose = detail::make_transpose_fn<I...>{};
