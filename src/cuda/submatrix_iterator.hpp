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
                           const int (&ub)[N])
{

    int current[] = {(0 * I)...};
    int n[] = {(ub[I] - lb[I] + 1)...};
    int stride[] = {stride_dim<I, N>(sz)...};

    if constexpr (N > sizeof...(I)) {

        return matrix_traversal_iterator<Iter, sizeof...(I), Iter, Iter>{
            it + ravel<N>(sz, lb), stride, current, n};

    } else if constexpr (N == sizeof...(I)) {

        int stride[] = {stride_dim<I, N>(sz)...};

        return matrix_traversal_iterator<Iter, N>{
            it + ravel<N>(sz, lb), stride, current, n};
    } else {
        static_assert(true, "something has gone terribly wrong");
    }
}
} // namespace detail

// for some reason, nvcc can't deduce N if we write `auto N`...
template <typename Iter, int N>
matrix_traversal_iterator<Iter, N>
make_submatrix(Iter it, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N])
{
    return detail::make_submatrix_helper<N>(
        std::make_index_sequence<N>{}, it, sz, lb, ub);
}
