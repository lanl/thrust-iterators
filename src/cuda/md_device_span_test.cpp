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

template struct md_device_span_cuda<double>;
