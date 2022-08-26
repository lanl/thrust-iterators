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

#include "traits.hpp"

#include "forward_stencil_iterator.hpp"
#include "thrust/iterator/zip_iterator.h"
#include <thrust/device_ptr.h>

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <type_traits>

namespace lazy
{

// simple structs with a templated call operator because we cannot annotate our lambdas
// with __host__ __device__ without special compiler flags
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

//
// apply_left/right for the case when a numeric value is involved
//
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

//
// apply_zip is used when we are operating on two iterators
//
template <typename Op>
struct apply_zip {
    Op op;

    template <typename Tp>
    __host__ __device__ constexpr auto operator()(Tp&& tp)
    {
        return op(thrust::get<0>(tp), thrust::get<1>(tp));
    }
};

//
// The transform_op struct is callable object returned by the lazy math operations.  When
// called, it forwards its arguments to its member(s) and returns a transform_iterator.
// Thus our iter_math leads to transform_ops of transform_ops of ... and a similar
// chaining of transform_iterators.
//
// To prevent unneccessary copies, the value category of the members
// of transform_op are functions of the input to the iter_math operators.  For
// example, if operation+ involves and lvalue& of lazy_vec<...>, then one of the members
// will be an lvalue& of lazy_vec<...>.  Rvalue refererences become values.  All
// arithmetic values are also captured by value
//
// This does conflate value category with lifetime but that shouldn't be a problem for the
// designed use
//
template <typename, typename, typename>
struct transform_op;

template <typename T>
struct iter_math {
private:
    // can't use auto for the return type since the functions then have the same
    // signature, triggering a redeclaration error
#define LAZY_VEC_OPERATORS(op, nextOp)                                                   \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<is_similar_v<T, U> && !is_stencil_proxy_v<V>>> \
    constexpr friend transform_op<boost::copy_cv_ref_t<T, U>,                            \
                                  arithmetic_by_value_t<V>,                              \
                                  nextOp>                                                \
    op(U&& u, V&& v)                                                                     \
    {                                                                                    \
        return {FWD(u), FWD(v)};                                                         \
    }                                                                                    \
                                                                                         \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<is_number_v<U> && is_similar_v<T, V>>>         \
    constexpr friend transform_op<arithmetic_by_value_t<U>,                              \
                                  boost::copy_cv_ref_t<T, V>,                            \
                                  nextOp>                                                \
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
struct transform_op : private iter_math<transform_op<U, V, Op>> {
    U u;
    V v;
    Op op;

    transform_op(U u, V v, Op op = {}) : u(u), v(v), op{op} {}

    template <typename... Bnds>
    constexpr auto operator()(Bnds&&... bnds)
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
};

} // namespace lazy
