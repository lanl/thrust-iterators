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

#include "traits.hpp"
#include <thrust/tuple.h>
#include <utility>

template <typename T>
static constexpr auto tp_size = thrust::tuple_size<un_cvref_t<T>>::value;

// merge tuples
namespace detail
{
template <auto... Is, typename L, auto... Js, typename R>
inline auto
merge_tuples_h(std::index_sequence<Is...>, L&& l, std::index_sequence<Js...>, R&& r)
{
    return thrust::make_tuple(thrust::get<Is>(FWD(l))..., thrust::get<Js>(FWD(r))...);
}
} // namespace detail

template <typename L, typename R>
inline decltype(auto) merge_tuples(L&& l, R&& r)
{
    if constexpr (tp_size<L> == 0 && tp_size<R> == 0)
        return FWD(l);
    else if constexpr (tp_size<L> == 0)
        return FWD(r);
    else if constexpr (tp_size<R> == 0)
        return FWD(l);
    else
        return detail::merge_tuples_h(std::make_index_sequence<tp_size<L>>{},
                                      FWD(l),
                                      std::make_index_sequence<tp_size<R>>{},
                                      FWD(r));
}

//
// append to tuples
//
namespace detail
{
template <auto... Is, typename Tp, typename U>
inline auto append_to_tuple_h(std::index_sequence<Is...>, Tp&& tp, U&& u)
{
    return thrust::make_tuple(thrust::get<Is>(FWD(tp))..., FWD(u));
}
} // namespace detail
template <typename Tp, typename U>
inline auto append_to_tuple(Tp&& tp, U&& u)
{
    return detail::append_to_tuple_h(
        std::make_index_sequence<tp_size<Tp>>{}, FWD(tp), FWD(u));
}

//
// prepending to tuples
//
namespace detail
{
template <auto... Is, typename Tp, typename U>
inline auto prepend_to_tuple_h(std::index_sequence<Is...>, U&& u, Tp&& tp)
{
    return thrust::make_tuple(FWD(u), thrust::get<Is>(FWD(tp))...);
}
} // namespace detail
template <typename U, typename Tp>
inline auto prepend_to_tuple(U&& u, Tp&& tp)
{
    return detail::prepend_to_tuple_h(
        std::make_index_sequence<tp_size<Tp>>{}, FWD(u), FWD(tp));
}
