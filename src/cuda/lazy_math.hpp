#pragma once

#include "forward_stencil_iterator.hpp"
#include "thrust/iterator/zip_iterator.h"
#include "traits.hpp"
#include <boost/type_traits/copy_cv_ref.hpp>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>

namespace lazy
{

struct plus {
    template <typename U, typename V>
    __host__ __device__ constexpr auto operator()(U&& u, V&& v) const
    {
        return u + v;
    }
};

struct multiplies {
    template <typename U, typename V>
    __host__ __device__ constexpr auto operator()(U&& u, V&& v) const
    {
        return u * v;
    }
};

struct minus {
    template <typename U, typename V>
    __host__ __device__ constexpr auto operator()(U&& u, V&& v) const
    {
        return u - v;
    }
};

struct divides {
    template <typename U, typename V>
    __host__ __device__ constexpr auto operator()(U&& u, V&& v) const
    {
        return u / v;
    }
};

struct gradient {
    template <typename H, typename It>
    __host__ __device__ constexpr auto operator()(H h, const stencil_t<It>& v) const
    {
        auto&& [x0, x1] = v;
        return (x1 - x0) / h;
    }
};

template <typename T, typename Op>
struct apply_right {
    T t;
    Op op;

    template <typename U>
    __host__ __device__ constexpr auto operator()(U&& u)
    {
        return op(t, FWD(u));
    }
};

template <typename T, typename Op>
struct apply_left {
    T t;
    Op op;

    template <typename U>
    __host__ __device__ constexpr auto operator()(U&& u)
    {
        return op(FWD(u), t);
    }
};

template <typename Op>
struct apply_zip {
    Op op;

    template <typename Tp>
    __host__ __device__ constexpr auto operator()(Tp&& tp)
    {
        return op(thrust::get<0>(tp), thrust::get<1>(tp));
    }
};

template <typename, auto...>
class lazy_vector;

template <typename>
struct lazy_vec_math;

class vec_math_access
{
    template <typename T>
    friend class lazy_vec_math;
};

template <typename, typename, typename>
struct transform_op;

template <typename T>
struct lazy_vec_math {
private:
    // can't use auto for the return type since the functions then have the same
    // signature, triggering a redeclaration error

#define LAZY_VEC_OPERATORS(op, nextOp)                                                   \
    template <typename U, typename V, typename = std::enable_if_t<is_similar_v<T, U>>>   \
    __host__ __device__ constexpr friend transform_op<boost::copy_cv_ref_t<T, U>,        \
                                                      arithmetic_by_value_t<V>,          \
                                                      nextOp>                            \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return {FWD(u), FWD(v)};                                                         \
    }                                                                                    \
                                                                                         \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<is_number_v<U> && is_similar_v<T, V>>>         \
    __host__ __device__ constexpr friend transform_op<arithmetic_by_value_t<U>,          \
                                                      boost::copy_cv_ref_t<T, V>,        \
                                                      nextOp>                            \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return {FWD(u), FWD(v)};                                                         \
    }

    LAZY_VEC_OPERATORS(operator*, multiplies)
    LAZY_VEC_OPERATORS(operator+, plus)
    LAZY_VEC_OPERATORS(operator-, minus)
    LAZY_VEC_OPERATORS(operator/, divides)

#undef LAZY_VEC_OPERATORS
};

template <typename U, typename V, typename Op>
struct transform_op : private lazy_vec_math<transform_op<U, V, Op>> {
    U u;
    V v;
    Op op;

    transform_op(U u, V v, Op op = {}) : u(u), v(v), op{op} {}

    template <typename... Bnds>
    __host__ __device__ constexpr auto operator()(Bnds&&... bnds)
    {
        if constexpr (std::is_arithmetic_v<U>)
            return thrust::make_transform_iterator(v(FWD(bnds)...),
                                                   apply_right<U, Op>{u, op});
        else if constexpr (std::is_arithmetic_v<V>)
            return thrust::make_transform_iterator(u(FWD(bnds)...),
                                                   apply_left<V, Op>{v, op});
        else
            return thrust::make_transform_iterator(
                thrust::make_zip_iterator(u(bnds...), v(bnds...)), apply_zip<Op>{op});
    }

private:
    friend class vec_math_access;
};

} // namespace lazy
