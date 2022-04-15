#pragma once

#include <boost/mp11.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <thrust/tuple.h>

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

// c++20 std::remove_cvref_t
template <typename T>
using un_cvref_t = typename detail::un_cvref<T>::type;

//
// traits for number - similar to is_arithmetic but is also true for references to
// arithmetic types since we generally do not care about references to double and so forth
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
// traits for dir_bounds
//
namespace lazy
{
template <auto>
struct dir_bounds;
}

namespace detail
{

template <typename T>
struct is_dir_bounds : std::false_type {
};

template <auto I>
struct is_dir_bounds<lazy::dir_bounds<I>> : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_dir_bounds_v = detail::is_dir_bounds<un_cvref_t<T>>::value;

//
// trait for lazy_vec_math
//
template <typename, auto...>
struct lazy_vector;

namespace lazy
{
template <typename>
struct lazy_vec_math;
}

namespace detail
{
template <typename T>
struct is_lazy_vec_math : std::is_base_of<lazy::lazy_vec_math<T>, T> {
};
} // namespace detail

template <typename T>
static constexpr auto is_lazy_vec_math_v = detail::is_lazy_vec_math<un_cvref_t<T>>::value;

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
// traits for stencil_assign_proxy
//
namespace lazy
{
template <auto, typename>
struct stencil_assign_proxy;
}

namespace detail
{
template <typename>
struct is_rhs_number : std::false_type {
};

template <auto N, typename T>
struct is_rhs_number<lazy::stencil_assign_proxy<N, T>>
    : std::is_arithmetic<un_cvref_t<T>> {
};
} // namespace detail

template <typename T>
static constexpr auto is_rhs_number_v = detail::is_rhs_number<un_cvref_t<T>>::value;

//
// traits for stencil_proxy -> needed for capturing base case of stencil_proxy<1>
//
namespace lazy
{
template <int N, typename = void, typename = void, typename = void>
struct stencil_proxy;
}

namespace detail
{
template <typename>
struct is_base_stencil_proxy : std::false_type {
};

template <>
struct is_base_stencil_proxy<lazy::stencil_proxy<0>> : std::true_type {
};

template <typename>
struct is_stencil_proxy : std::false_type {
};

template <int N, typename U, typename V, typename Op>
struct is_stencil_proxy<lazy::stencil_proxy<N, U, V, Op>> : std::true_type {
};

template <typename>
struct proxy_index;

template <int N, typename U, typename V, typename Op>
struct proxy_index<lazy::stencil_proxy<N, U, V, Op>> {
    static constexpr int value = N;
};
} // namespace detail

template <typename T>
static constexpr auto is_base_stencil_proxy_v =
    detail::is_base_stencil_proxy<un_cvref_t<T>>::value;

template <typename T>
static constexpr auto is_stencil_proxy_v = detail::is_stencil_proxy<un_cvref_t<T>>::value;

template <typename T>
static constexpr auto proxy_index_v = detail::proxy_index<un_cvref_t<T>>::value;

//
// combining proxies in our system requires enforcing certain ground rules
// 1. Baseline stencil_proxy start with an index (first template parameter) of 0
// 2. Combining 2 proxyies with 0 index results in a 0 index
// 3. Combining a proxy with index `N` and a number results in a `N` index
// 4. Combining a proxy with index `N` and a lazy_vec results in `N+1` index
// 5. Combining proxies with 0 and `N` index results in an `N` index
// 6. Combining 2 proxies with non-zero index is an error
//

namespace detail
{
template <typename U, typename V>
struct next_proxy_index {
    static_assert(is_stencil_proxy_v<U> || is_stencil_proxy_v<V>);

    static constexpr auto index()
    {
        if constexpr (is_stencil_proxy_v<U>) {
            constexpr auto u = proxy_index_v<U>;
            if constexpr (is_stencil_proxy_v<V>) {
                constexpr auto v = proxy_index_v<V>;
                return v > u ? v : u;
            } else if constexpr (is_number_v<V>) {
                return u;
            } else if constexpr (is_lazy_vec_math_v<V>) {
                return u + 1;
            } else if constexpr (true) {
                static_assert(true, "how did we get here?");
            }
        } else {
            constexpr auto v = proxy_index_v<V>;
            if constexpr (is_number_v<U>) {
                return v;
            } else if constexpr (is_lazy_vec_math_v<U>) {
                return v + 1;
            } else if constexpr (true) {
                static_assert(true, "how did we get here?");
            }
        }
    }
};

} // namespace detail

template <typename X, typename Y>
static constexpr int
    next_proxy_index_v = detail::next_proxy_index<un_cvref_t<X>, un_cvref_t<Y>>::index();

//
// traits for self_assign_proxy
//
namespace lazy
{
template <typename, typename>
struct self_assign_proxy;
}

namespace detail
{
template <typename T>
struct is_self_assign_proxy : std::false_type {
};

template <typename U, typename V>
struct is_self_assign_proxy<lazy::self_assign_proxy<U, V>> : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_self_assign_proxy_v =
    detail::is_self_assign_proxy<un_cvref_t<T>>::value;

//
// traits for ensuring arithmetic values are by-value for transform_op
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
// Compile time index manipulation to facilitate invisible transpose operations
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
// Utilities for finding the "missing" dimension for coarse_to_fine
//
template <typename Fine, typename Coarse>
static constexpr auto missing_index_v =
    mp::mp_front<mp::mp_set_difference<Fine, Coarse>>::value;

//
// Utilities for computing the up/down shift needed for gradients
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

//
// trait for identifying thrust tuple or pair
//
namespace detail
{
template <typename, typename = std::void_t<>>
struct is_tuple : std::false_type {
};

template <typename T>
struct is_tuple<T, std::void_t<decltype(thrust::get<0>(std::declval<T>()))>>
    : std::true_type {
};
} // namespace detail

template <typename T>
static constexpr auto is_tuple_v = detail::is_tuple<un_cvref_t<T>>::value;
