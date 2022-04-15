#pragma once

#include "traits.hpp"
#include "tuple_utils.hpp"
#include "lazy_math.hpp"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <type_traits>

namespace lazy
{

template <typename T>
struct lazy_proxy_math {
private:
#define LAZY_VEC_OPERATORS(op, nextOp)                                                   \
    template <typename U, typename V, typename = std::enable_if_t<is_similar_v<T, U>>>   \
    constexpr friend stencil_proxy<next_proxy_index_v<U, V>,                             \
                                   boost::copy_cv_ref_t<T, U>,                           \
                                   arithmetic_by_value_t<V>,                             \
                                   nextOp>                                               \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return {FWD(u), FWD(v)};                                                         \
    }                                                                                    \
                                                                                         \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<(is_number_v<U> ||                             \
                                           is_lazy_vec_math_v<U>)&&is_similar_v<T, V>>>  \
    constexpr friend stencil_proxy<next_proxy_index_v<U, V>,                             \
                                   arithmetic_by_value_t<U>,                             \
                                   boost::copy_cv_ref_t<T, V>,                           \
                                   nextOp>                                               \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return {FWD(u), FWD(v)};                                                         \
    }

    LAZY_VEC_OPERATORS(operator*, multiplies)
    LAZY_VEC_OPERATORS(operator+, plus)
    LAZY_VEC_OPERATORS(operator-, minus)
    LAZY_VEC_OPERATORS(operator/, divides)

#undef LAZY_VEC_OPERATORS

    template <typename U, typename = std::enable_if_t<is_similar_v<T, U>>>
    constexpr friend stencil_proxy<proxy_index_v<U>,
                                   int,
                                   boost::copy_cv_ref_t<T, U>,
                                   multiplies>
    operator-(U&& u)
    {
        return {-1, FWD(u)};
    }
};

namespace detail
{
template <int N, typename U, typename It, typename T>
__host__ __device__ decltype(auto) maybe_invoke_with(U&& u, It&& it, T&& t)
{
    if constexpr (is_number_v<U>) {
        return u;
    } else if constexpr (is_base_stencil_proxy_v<U>) {
        return u(FWD(it));
    } else if constexpr (is_lazy_vec_math_v<U>) {
        return thrust::get<N - 1>(t);
    } else {
        return u(FWD(it), FWD(t));
    }
}
} // namespace detail

template <int N, typename U, typename V, typename Op>
struct stencil_proxy : lazy_proxy_math<stencil_proxy<N, U, V, Op>> {
    U u;
    V v;
    Op op;

    stencil_proxy(U u, V v, Op op = {}) : u{FWD(u)}, v{FWD(v)}, op{MOVE(op)} {}

    template <typename It, typename T, typename = std::enable_if_t<!is_dir_bounds_v<It>>>
    __host__ __device__ decltype(auto) operator()(It&& it, T&& t)
    {
        return op(detail::maybe_invoke_with<N>(u, it, t),
                  detail::maybe_invoke_with<N>(v, it, t));
    }

    template <typename... P, typename = std::enable_if_t<(is_dir_bounds_v<P> && ...)>>
    auto operator()(P&&... bnds)
    {
        if constexpr (N == 0) {
            return thrust::make_zip_iterator(
                thrust::make_tuple(thrust::constant_iterator<int>(0)));
        } else {
            return thrust::make_zip_iterator(get_iterators(FWD(bnds)...));
        }
    }

    template <typename... P>
    auto get_iterators(P&&... bnds)
    {
        if constexpr (is_stencil_proxy_v<U>) {
            if constexpr (is_stencil_proxy_v<V>)
                return merge_tuples(u.get_iterators(bnds...), v.get_iterators(bnds...));
            else if constexpr (is_number_v<V>)
                return u.get_iterators(bnds...);
            else if constexpr (is_lazy_vec_math_v<V>)
                return append_to_tuple(u.get_iterators(bnds...), v(bnds...));
            else
                static_assert(true, "we should never get here");
        } else if constexpr (is_number_v<U>) {
            if constexpr (is_stencil_proxy_v<V>)
                return v.get_iterators(bnds...);
            else if constexpr (is_number_v<V>)
                return thrust::tuple<>{};
            else if constexpr (is_lazy_vec_math_v<V>)
                return thrust::make_tuple(v(bnds...));
            else
                static_assert(true, "we should never get here");
        } else if constexpr (is_lazy_vec_math_v<U>) {
            if constexpr (is_stencil_proxy_v<V>)
                return prepend_to_tuple(u(bnds...), v.get_iterators(bnds...));
            else if constexpr (is_number_v<V>)
                return thrust::make_tuple(u(bnds...));
            else if constexpr (is_lazy_vec_math_v<V>) // I don't think this can happen
                return thrust::make_tuple(u(bnds...), v(bnds...));
            else
                static_assert(true, "we should never get here");
        } else
            static_assert(true, "we should never get here");
    }
};

template <>
struct stencil_proxy<0> : lazy_proxy_math<stencil_proxy<0>> {
    int i;

    stencil_proxy(int i) : i{i} {}

    template <typename T>
    stencil_assign_proxy<1, T> operator=(T&& t)
    {
        return {i, FWD(t)};
    }

    template <typename It>
    __host__ __device__ decltype(auto) operator()(It&& it) const
    {
        return FWD(it)[i];
    }

    template <typename... P>
    auto get_iterators(P&&...)
    {
        return thrust::tuple<>{};
    }
};

// In general, the proxy with two indices is not part of a larger more complicated
// expression so we wont be checking for this case (and could technically have made the
// '2' anything)
template <>
struct stencil_proxy<1> {
    int i, j;

    template <typename T>
    stencil_assign_proxy<2, T> operator=(T&& t)
    {
        return {i, j, FWD(t)};
    }
};

//
// computing stencil coefficients makes heavy use of invokables.  Store them in a tuple
// for ease of use
//
template <typename... Args>
struct tp_invoke {
    thrust::tuple<Args...> t;

    tp_invoke(Args&&... args) : t(thrust::make_tuple(FWD(args)...)) {}

    template <typename It>
    __host__ __device__ void operator()(It&& it)
    {
        if constexpr (is_tuple_v<It>) {
            static_assert(thrust::tuple_size<un_cvref_t<It>>::value ==
                          1 + sizeof...(Args));
            (*this)(thrust::get<0>(it), it, std::make_index_sequence<sizeof...(Args)>{});
        } else {
            (*this)(FWD(it), std::make_index_sequence<sizeof...(Args)>{});
        }
    }

    template <typename It, auto... I>
    __host__ __device__ void operator()(It&& it, std::index_sequence<I...>)
    {
        (thrust::get<I>(t)(it), ...);
    }

    template <typename It, typename Tp, auto... I>
    __host__ __device__ void operator()(It&& it, Tp&& tp, std::index_sequence<I...>)
    {
        (thrust::get<I>(t)(it, thrust::get<I + 1>(tp)), ...);
    }
};

template <typename... Args>
tp_invoke(Args&&...) -> tp_invoke<Args...>;

template <typename... Args>
struct tp_layered_invoke {
    thrust::tuple<Args...> t;

    tp_layered_invoke(Args&&... args) : t(thrust::make_tuple(FWD(args)...)) {}

    template <typename It>
    __host__ __device__ void operator()(It&& it)
    {
        static_assert(is_tuple_v<It>);
        static_assert(thrust::tuple_size<un_cvref_t<It>>::value == 2);
        (*this)(thrust::get<0>(FWD(it)),
                thrust::get<1>(FWD(it)),
                std::make_index_sequence<sizeof...(Args)>{});
    }

    template <typename L, typename R, auto... I>
    __host__ __device__ void operator()(L&& l, R&& r, std::index_sequence<I...>)
    {
        static_assert(is_tuple_v<L>);
        static_assert(is_tuple_v<R>);
        static_assert(thrust::tuple_size<un_cvref_t<L>>::value + 1 ==
                      thrust::tuple_size<un_cvref_t<R>>::value);
        (thrust::get<I>(t)(
             thrust::get<I>(FWD(l)), thrust::get<0>(FWD(r)), thrust::get<I + 1>(FWD(r))),
         ...);
    }
};

template <typename... Args>
tp_layered_invoke(Args&&...) -> tp_layered_invoke<Args...>;

template <typename T>
struct stencil_assign_proxy<1, T> {
    int i;
    T t;

    template <typename It>
    __host__ __device__ void operator()(It&& it)
    {
        it[i] = t;
    }

    template <typename It, typename U>
    __host__ __device__ void operator()(It&& it, U&& u)
    {

        if constexpr (is_lazy_vec_math_v<T>)
            it[i] = u;
        else if constexpr (is_number_v<T>)
            it[i] = t;
        else
            // here we assume that t is a stencil_proxy.  Should make this more robust
            it[i] = t(it, FWD(u));
    }

    template <typename... P>
    auto get_iterator(P&&... p)
    {
        if constexpr (is_number_v<T>)
            return thrust::constant_iterator<int>(0);
        else
            return t(FWD(p)...);
    }
};

template <typename T>
struct stencil_assign_proxy<2, T> {
    int i, j;
    T t;

    template <typename It>
    __host__ __device__ void operator()(It&& it)
    {
        if constexpr (is_number_v<T>) {
            it[i] = t;
            it[j] = t;
        } else {
            static_assert(true);
        }
    }

    template <typename It, typename U>
    __host__ __device__ void operator()(It&& it, U&& u)
    {
        // for now we assume that U is a stencil_t.  This should be made more robust in
        // the future
        it[i] = u.a;
        it[j] = u.b;
    }

    template <typename... P>
    auto get_iterator(P&&... p)
    {
        if constexpr (is_number_v<T>)
            return thrust::constant_iterator<int>(0);
        else
            return t(FWD(p)...);
    }
};

template <typename L, typename R>
struct self_assign_proxy {
    L l;
    R r;

    static_assert(is_stencil_proxy_v<R>);

    self_assign_proxy(L&& l, R&& r) : l{FWD(l)}, r{FWD(r)} {}

    template <typename It_l, typename W, typename It_r>
    __host__ __device__ void operator()(It_l&& it_l, W&& w, It_r&& it_r)
    {
        it_l = r(FWD(w), FWD(it_r));
    }

    template <typename... P>
    auto get_lhs_iterator(P&&... p)
    {
        return l(FWD(p)...);
    }

    template <typename... P>
    auto get_iterator(P&&... p)
    {
        return r(FWD(p)...);
    }
};
} // namespace lazy
