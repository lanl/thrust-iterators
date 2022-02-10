#pragma once

#include "matrix_traversal_iterator.hpp"


namespace detail
{
template <typename Iter, auto N, auto... I>
matrix_traversal_iterator<Iter, N> make_submatrix_helper(std::index_sequence<I...>,
                                                         Iter it,
                                                         const int (&sz)[N],
                                                         const int (&lb)[N],
                                                         const int (&ub)[N])
{
    int stride[] = {stride_dim<I, N>(sz)...};
    int current[] = {(0 * I)...};
    int n[] = {(ub[I] - lb[I] + 1)...};

    return {it + ravel<N>(sz, lb), stride, current, n};
}
} // namespace detail

// for some reason, nvcc can't deduce N if we write `auto N`...
template <typename Iter, int N>
matrix_traversal_iterator<Iter, N>
make_submatrix(Iter it, const int (&sz)[N], const int (&lb)[N], const int (&ub)[N])
{
    return detail::make_submatrix_helper<Iter, N>(std::make_index_sequence<N>{}, it, sz, lb, ub);
}
