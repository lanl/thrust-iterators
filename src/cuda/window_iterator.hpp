#pragma once

#include "matrix_traversal_iterator.hpp"
#include "md_device_span.hpp"
#include "traits.hpp"

namespace detail
{

template <auto N, typename Iter, size_t... I>
matrix_traversal_iterator<Iter, N, Iter, Iter>
window_helper(std::index_sequence<I...>, Iter it, const int (&sz)[N + 1])
{
    int stride[] = {stride_dim<I, N + 1>(sz)...};
    int current[] = {(0 * I)...};
    int n[] = {sz[I]...};
    int stm[] = {(0 * I + 1)...};

    return {it, stride, current, n, stm};
}

} // namespace detail

// construct a multidimensional window iterator from an ND lazy_vec assuming the last
// dimension is the window size
template <typename T, auto... Order>
auto window(md_device_span<T, Order...>& v)
{
    static constexpr auto N = v.N - 1;
    auto db = v.dir_bounds();
    // this atrocious construct is because we can't return c-arrays...
    int sz[] = {std::get<map_index_v<index_list<Order...>, Order>>(db).size()...};

    return detail::window_helper<N>(std::make_index_sequence<N>{}, v.begin(), sz);
}
