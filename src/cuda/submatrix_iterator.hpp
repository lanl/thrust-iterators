#pragma once

#include "matrix_traversal_iterator.hpp"
#include <utility>

namespace detail
{
template <auto N, typename Iter, size_t... I>
auto make_submatrix_helper(std::index_sequence<I...>,
                           Iter it,
                           const int (&sz)[N],
                           const int (&lb)[N],
                           const int (&ub)[N],
                           const int (&stm)[N])
{


    int current[] = {(0 * I)...};
    int n[] = {((ub[I] - lb[I] + stm[I]) / stm[I])...};
    int stride[] = {stride_dim<I, N>(sz, stm)...};

    // printf("submatrix helper with\n");
    // printf("\tsz\t%i", sz[0]);
    // for (int i = 1; i < N; i++) printf("\t%i", sz[i]);
    // printf("\n\tstm\t%i", stm[0]);
    // for (int i = 1; i < N; i++) printf("\t%i", stm[i]);
    // printf("\n\tstride\t%i", stride[0]);
    // for (int i = 1; i < sizeof...(I); i++) printf("\t%i", stride[i]);
    // printf("\n");

    if constexpr (N > sizeof...(I)) {

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
