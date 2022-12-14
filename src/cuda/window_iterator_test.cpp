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



#include "md_device_span.hpp"
#include "traits.hpp"
#include "window_iterator.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/set.hpp>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <utility>

#include "../window_iterator_test.hpp"

template <typename T>
void window_test_cuda<T>::init(T* v_, int n)
{
    assert(n == 10);
    const auto i = Ib{1, 2};
    const auto w = Wb{0, 4};

    auto v = make_md_span(v_, i, w);

    auto st = v.window()(i);
    auto it = *st;
    *it = 1.0;
    it[1] = 2.0;

    auto f0 = (v(0) = 3.0);
    auto f1 = (v(1) = 4.0);

    ++st;
    f0(*st);
    f1(*st);
}

template <typename T>
void window_test_cuda<T>::transform(T* v_, int n)
{
    assert(n == 10);
    const auto i = Ib{1, 5};
    const auto w = Wb{0, 1};

    auto v = make_md_span(v_, i, w);

    with_domain(v.window(), i)(v(0) = 1.0, v(1) = v(1) + 2.0);
}

template <typename T>
void window_test_cuda<T>::transform2(T* v_, int n)
{
    assert(n == 15);
    const auto i = Ib{1, 5};
    const auto w = Wb{0, 2};

    auto v = make_md_span(v_, i, w);
    T x = 4;

    with_domain(v.window(), i)(v(0) = -(v(1) + v(2)) + x - 1);
}

template <typename T>
void window_test_cuda<T>::transform3(T* v_, int n, const T* u_)
{
    assert(n == 15);
    const auto i = Ib{1, 5};
    const auto w = Wb{0, 2};

    auto v = make_md_span(v_, i, w);
    auto u = make_md_span(u_, i);
    T x = 4;

    auto f = -(v(1) + v(2)) + 3 * u;
    using X = lazy::stencil_proxy<0>;
    static_assert(std::is_same_v<X, decltype(v(1))>);
    using Y = lazy::stencil_proxy<0, X, X, lazy::plus>;
    static_assert(std::is_same_v<Y, decltype(v(1) + v(2))>);
    using YY = lazy::stencil_proxy<0, int, Y, lazy::multiplies>;
    static_assert(std::is_same_v<YY, decltype(-(v(1) + v(2)))>);

    using V = md_device_span<const T, lazy::dim::I>;
    static_assert(std::is_same_v<V, decltype(u)>);
    using Z = lazy::transform_op<int, V&, lazy::multiplies>;
    static_assert(std::is_same_v<Z, decltype(3 * u)>);

    static_assert(std::is_same_v<lazy::stencil_proxy<1, YY, Z, lazy::plus>, decltype(f)>);

    auto it = f(i);
    static_assert(tp_size<decltype(*it)> == 1);
    T u3 = thrust::get<0>(*it);
    with_domain(v.window(), i)(v(0) = -(v(1) + v(2)) + 3 * u);
}

template <typename T>
void window_test_cuda<T>::transform4(T* v_, int n, const T* u_)
{
    assert(n == 20);
    //const auto i0 = Ib{1, 5, 2};
    const auto i = Ib{1, 5, 1};
    const auto w = Wb{0, 3};

    auto v = make_md_span(v_, i, w);
    auto u = make_md_span(u_, i);
    T x = 4;

    auto f = 3 * u;

    with_domain(v.window(), i)(v(0) = -(v(2) + v(3)) - f, v(1) = 2 * f);
}

template <typename T>
void window_test_cuda<T>::rhs(T* rhs_, const T* v_, int n, const T* u_)
{

    assert(n == 20);
    const auto i = Ib{1, 5};
    const auto w = Wb{0, 3};

    auto rhs = make_md_span(rhs_, i);
    auto v = make_md_span(v_, i, w);
    auto u = make_md_span(u_, i);

    auto c = (rhs -= v(1) * u);
    static_assert(is_self_assign_proxy_v<decltype(c)>);

    auto d = (rhs - v(1) * u);
    static_assert(is_stencil_proxy_v<decltype(d)>);

    with_domain(v.window(), i)(rhs -= v(1) * u);

    using L = index_list<lazy::dim::K, lazy::dim::J, lazy::dim::I>;
    using M = index_list<lazy::dim::K, lazy::dim::J>;
    static_assert(missing_index_v<L, M> == lazy::dim::I);

    using N = index_list<lazy::dim::K, lazy::dim::I>;
    static_assert(missing_index_v<L, N> == lazy::dim::J);
}

template struct window_test_cuda<double>;
