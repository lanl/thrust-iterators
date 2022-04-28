#pragma once

#include "traits.hpp"

#include <thrust/tuple.h>
#include <type_traits>

template <typename T>
struct stencil_t {
    T a, b;

private:
#define LAZY_VEC_OPERATORS(op, infixOp)                                                  \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<is_similar_v<stencil_t, U> && is_number_v<V>>> \
    __host__ __device__ constexpr friend stencil_t<                                      \
        decltype(std::declval<T>() infixOp std::declval<V>())>                           \
    op(U&& u, V v)                                                                       \
    {                                                                                    \
        return {FWD(u).a infixOp v, FWD(u).b infixOp v};                                 \
    }                                                                                    \
                                                                                         \
    template <typename U,                                                                \
              typename V,                                                                \
              typename = std::enable_if_t<is_number_v<U> && is_similar_v<stencil_t, V>>> \
    constexpr friend stencil_t<decltype(std::declval<U>() infixOp std::declval<T>())>    \
        __host__ __device__ op(U u, V&& v)                                               \
    {                                                                                    \
        return {u infixOp FWD(v).a, u infixOp FWD(v).b};                                 \
    }
    LAZY_VEC_OPERATORS(operator+, +)
    LAZY_VEC_OPERATORS(operator*, *)
    LAZY_VEC_OPERATORS(operator-, -)
    LAZY_VEC_OPERATORS(operator/, /)

#undef LAZY_VEC_OPERATORS
};
