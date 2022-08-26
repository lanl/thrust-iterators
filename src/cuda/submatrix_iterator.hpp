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

#include "matrix_traversal_iterator.hpp"
#include <type_traits>
#include <utility>

namespace detail
{
template <auto N, typename Iter, size_t... I, typename T = std::false_type>
auto make_submatrix_helper(std::index_sequence<I...>,
                           Iter it,
                           const int (&sz)[N],
                           const int (&lb)[N],
                           const int (&ub)[N],
                           const int (&stm)[N],
                           T t = {})
{


    int current[] = {(0 * I)...};
    int n[] = {((ub[I] - lb[I] + stm[I]) / stm[I])...};
    int stride[] = {stride_dim<I, N>(sz, stm)...};

    if constexpr (N > sizeof...(I) || std::is_same_v<T, std::true_type>) {

        return matrix_traversal_iterator<Iter, sizeof...(I), Iter, Iter>{
            it + ravel<N>(sz, lb), stride, current, n};

    } else if constexpr (N == sizeof...(I)) {

        return matrix_traversal_iterator<Iter, N>{
            it + ravel<N>(sz, lb), stride, current, n};
    } else {
        static_assert(true, "something has gone terribly wrong");
    }
}
} // namespace detail

// for some reason, nvcc can't deduce N if we write `auto N`...
template <typename Iter, int N>
matrix_traversal_iterator<Iter, N> make_submatrix(Iter it,
                                                  const int (&sz)[N],
                                                  const int (&lb)[N],
                                                  const int (&ub)[N],
                                                  const int (&stm)[N])
{
    return detail::make_submatrix_helper<N>(
        std::make_index_sequence<N>{}, it, sz, lb, ub, stm);
}
