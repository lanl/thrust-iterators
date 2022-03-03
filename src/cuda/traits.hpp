#pragma once

#include <boost/mp11.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

#define FWD(x) static_cast<decltype(x)&&>(x)
#define MOVE(x) static_cast<std::remove_reference_t<decltype(x)>&&>(x)

namespace mp = boost::mp11;

namespace detail
{
template <typename T>
struct un_cvref {
    using type = typename std::remove_cv_t<std::remove_reference_t<T>>;
};
} // namespace detail

template <typename T>
using un_cvref_t = typename detail::un_cvref<T>::type;

//
// traits for number - similar to is_arithmetic but is also true for references to
// arithmetic types
//
template <typename T>
static constexpr auto is_number_v = std::is_arithmetic_v<un_cvref_t<T>>;

//
// traits for bounds type
//
struct bounds;

namespace detail
{
template <typename T>
struct is_bounds : std::false_type {
};

template <>
struct is_bounds<bounds> : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_bounds_v = detail::is_bounds<un_cvref_t<T>>::value;

//
// traits for bound_dim
//
namespace lazy
{
template <auto, typename>
struct bound_dim;
}

namespace detail
{

template <typename T>
struct is_bound_dim : std::false_type {
};

template <auto I, typename T>
struct is_bound_dim<lazy::bound_dim<I, T>> : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_bound_dim_v = detail::is_bound_dim<un_cvref_t<T>>::value;

//
// traits for assign_proxy
//
namespace lazy
{
template <typename, typename>
struct assign_proxy;
}

namespace detail
{
template <typename T>
struct is_assign_proxy : std::false_type {
};

template <typename U, typename V>
struct is_assign_proxy<lazy::assign_proxy<U, V>> : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_assign_proxy_v = detail::is_assign_proxy<un_cvref_t<T>>::value;

//
// traits for ensuring arithmetic values are by-value
//
namespace detail
{
template <typename T, bool = std::is_arithmetic_v<un_cvref_t<T>>>
struct arithmetic_by_value_impl {
    using type = un_cvref_t<T>;
};

template <typename T>
struct arithmetic_by_value_impl<T, false> {
    using type = T;
};

} // namespace detail

template <typename T>
using arithmetic_by_value_t = typename detail::arithmetic_by_value_impl<T>::type;

//
// Compile time index manipulation
//

template <size_t... Is>
using index_list = mp::mp_list_c<size_t, Is...>;

template <typename L, typename I>
struct map_index : mp::mp_find<L, I> {
};

template <typename L, auto I>
struct map_index<L, mp::mp_size_t<I>> : mp::mp_find<L, mp::mp_size_t<I>> {
};

template <typename L, typename I>
using map_index_t = typename map_index<L, I>::type;

template <typename L, size_t I>
static constexpr auto map_index_v = map_index<L, mp::mp_size_t<I>>::value;

//
// convert from mp_list to index_sequence
//
template <typename...>
struct to_sequence;

template <typename T, auto... Is>
struct to_sequence<std::integral_constant<T, Is>...> {
    using type = std::index_sequence<Is...>;
};

template <typename... Ts>
struct to_sequence<mp::mp_list<Ts...>> {
    using type = typename to_sequence<Ts...>::type;
};

template <typename T>
using to_sequence_t = typename to_sequence<T>::type;

template <typename From, typename To>
struct transpose_sequence {
    using type =
        to_sequence_t<mp::mp_transform_q<mp::mp_bind_front<map_index_t, From>, To>>;
};

template <typename From, typename To>
using transpose_sequence_t = typename transpose_sequence<From, To>::type;

//
// Utilities for computing the shift needed for gradients
//
template <int Shift, auto X, auto Y>
struct select_shift : std::conditional<X == Y, mp::mp_int<Shift>, mp::mp_int<0>> {
};

template <int S, auto X, auto Y>
static constexpr int shift_v = select_shift<S, X, Y>::type::value;

//
// traits for same ignoring cvref qualifiers
//
template <typename T, typename U>
static constexpr auto is_similar_v = std::is_same_v<un_cvref_t<T>, un_cvref_t<U>>;
