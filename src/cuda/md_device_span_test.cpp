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


#include "../md_device_span_cuda.hpp"

#include "md_device_span.hpp"
#include "traits.hpp"
#include <type_traits>
#include <utility>

template <typename T>
void md_device_span_cuda<T>::init(const int& i0,
                                  const int& i1,
                                  const T& beta,
                                  const T* dx,
                                  const int ugcw,
                                  const T* u_,
                                  const int& rgcw,
                                  T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_md_span(u_, ugcw, i);
    auto res = make_md_span(res_, rgcw, i);

    with_domain(i)(res = 2 * (u.grad_x(dx[0], down) + u.grad_x(dx[0])) / (u + 10));

    // syntax check - should expand and move to anote
    auto it = (3 * u.stencil_x())(i);
    auto x = *it;
    T a = x.a;
    T b = x.b;
}

template <typename T>
void md_device_span_cuda<T>::stride(const int& i0,
                                    const int& i1,
                                    const int& stride,
                                    const T& beta,
                                    const T* dx,
                                    const int ugcw,
                                    const T* u_,
                                    const int& rgcw,
                                    T* res_)
{
    const auto i = Ib{i0, i1, stride};

    auto u = make_md_span(u_, ugcw, i);
    auto res = make_md_span(res_, rgcw, i);

    with_domain(i)(res = beta * u);
}

template <typename T>
void md_device_span_cuda<T>::init(const int& i0,
                                  const int& i1,
                                  const int& j0,
                                  const int& j1,
                                  const T& beta,
                                  const T* dx,
                                  const int ugcw,
                                  const T* u_,
                                  const int& rgcw,
                                  T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_md_span(u_, ugcw, i, j);
    auto res = make_md_span(res_, rgcw, j, i);

    with_domain(j, i)(res = 3 * (u.grad_y(dx[1]) + u.grad_x(dx[0])));

    using Seq = transpose_sequence_t<index_list<2, 0, 1, 3>, index_list<0, 1, 2>>;
    static_assert(std::is_same_v<Seq, std::index_sequence<1, 2, 0>>);

    static_assert(is_tuple_v<thrust::tuple<int>>);
    static_assert(is_tuple_v<thrust::pair<int, float>>);
    static_assert(!is_tuple_v<int>);

    static_assert(is_iter_math_v<decltype(u)>);
    static_assert(is_iter_math_v<decltype(u + 3)>);
}

template <typename T>
void md_device_span_cuda<T>::stride(const int& i0,
                                    const int& i1,
                                    const int& is,
                                    const int& j0,
                                    const int& j1,
                                    const int& js,
                                    const T& beta,
                                    const T* dx,
                                    const int ugcw,
                                    const T* u_,
                                    const int& rgcw,
                                    T* res_)
{
    const auto i = Ib{i0, i1, is};
    const auto j = Jb{j0, j1, js};

    auto u = make_md_span(u_, ugcw, i, j);
    auto res = make_md_span(res_, rgcw, j, i);

    with_domain(j, i)(res = beta * u);
}

template <typename T>
void md_device_span_cuda<T>::stride(const int& i0,
                                    const int& i1,
                                    const int& is,
                                    const int& j0,
                                    const int& j1,
                                    const int& js,
                                    const int& k0,
                                    const int& k1,
                                    const int& ks,
                                    const T& beta,
                                    const T* dx,
                                    const int ugcw,
                                    const T* u_,
                                    const int& rgcw,
                                    T* res_)
{
    const auto i = Ib{i0, i1, is};
    const auto j = Jb{j0, j1, js};
    const auto k = Kb{k0, k1, ks};

    auto u = make_md_span(u_, ugcw, i, j, k);
    auto res = make_md_span(res_, rgcw, k, j, i);

    with_domain(k, j, i)(res = beta * u);
}


template struct md_device_span_cuda<double>;
