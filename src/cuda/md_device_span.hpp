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

#include "iter_math.hpp"
#include "offset_iterator.hpp"
#include "stencil_proxy.hpp"
#include "thrust/for_each.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "traits.hpp"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <type_traits>
#include <utility>

#include "md_bounds.hpp"

#include "submatrix_iterator.hpp"
#include "coarse_to_fine_iterator.hpp"

//
// Machinery for a multidimensional lazy vector class.  The purpose is to allow us to
// write code that looks like math.  This also allows for composable operations without
// the penalty imposed by eager evaluation.  Note that we rely heavily on the compiler
// inlining the variety of nested functions to produce efficient code
//
//
// All the actual work is done by the matrix_traversal and forward_stencil iterators.  The
// mechanisims here are largely for ensuring the iterators are properly constructed and
// combined.
//

//
// Gradients in the `up` direction (the default) use indices [i,i+1].
// Gradients in the `down` direction use indices [i-1,i].
//
static constexpr auto up = mp::mp_int<1>{};
static constexpr auto down = mp::mp_int<-1>{};

//
// After constructing bounds i,j,k using Ib,Jb,Kb, the user should be able to intuitively
// describe the domain of the calculation with something like:
//
// with_domain(k, j, i)
//

namespace lazy
{
//
// callable object returned by the lazy vec assignment operator.  When invoked, this is
// the driver for the thrust computation.  This is implicitly invoked by the lambda
// returned by with_domain(..).  The proxy's members are captured by reference when
// feasible
//
template <typename U, typename V>
struct assign_proxy {
    U u;
    V v;

    template <typename... Bnds,
              typename = std::enable_if_t<(is_dir_bounds_v<Bnds> && ...)>>
    void operator()(Bnds&&... bnds)
    {
        auto out = u(bnds...);
        if constexpr (is_number_v<V>)
            thrust::fill_n(thrust::device, out, out.size(), v);
        else
            thrust::copy_n(thrust::device, v(FWD(bnds)...), out.size(), out);
    }
};

//
// Computing gradients is a bit tricker because we may need to transpose the data first.
// In the event that we transpose the data, we also need to change the direction of the
// gradient.  This allows the user to simply write, for example, grad_x to also refer to
// the gradient in the I direction, without figuring out which dimension will correspond
// to the I direction after any required transpose operations.
//
// However, we don't which direction the transpose will be until the user invokes the
// assign_proxy with the desired bounds.  Therefore, we need to store this calculation in
// a callable object that will do the right thing as a member of transform_op.
template <int Shift, typename T, int I>
struct stencil_helper : iter_math<stencil_helper<Shift, T, I>> {
    T t;

    stencil_helper(T&& t) : t{FWD(t)} {}

    template <int... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t((bnds + shift_v<Shift, I, O>)...)
            .template stencil<map_index_v<index_list<O...>, I>>();
    }
};

template <int Shift, int N, typename Vec, typename T>
transform_op<T, stencil_helper<Shift, Vec, N>, gradient>
make_gradient_transform(Vec&& vec, T h)
{
    return {h, {FWD(vec)}};
}

//
// Computing dot products requires that we store the range of indicies and the pointer to
// offsets.  We are here assuming that dot products are always done with "stencil" spans
// (where W is the last dim) and offset_helpers.
//
// Note that we manually flip the order of the strides here to match the reversed order of
// the offsets.  This is brittle and should be fixed at some point
//
template <int N>
struct dot_impl {
    int stride[N];
    thrust::device_ptr<int> offsets;
    int n;

    template <typename U>
    dot_impl(U&& u, thrust::device_ptr<int> o, int n) : offsets{o}, n{n}
    {
        for (int i = 0; i < N; i++) stride[N - 1 - i] = u.stride_dim(i);
    }

    template <typename Tp>
    __host__ __device__ auto operator()(Tp&& tp) const
    {
        auto&& st = thrust::get<0>(tp);
        auto&& it = thrust::get<1>(tp);
        auto o = make_offset_iterator(FWD(it), stride, offsets);
        using T = decltype(*st * *o);

        T res{};

        for (int i = 0; i < n; i++) res += st[i] * o[i];

        return res;
    }
};

template <typename U, typename V>
struct dot_helper : iter_math<dot_helper<U, V>> {
    U u;
    V v;

    dot_helper(U&& u, V&& v) : u{FWD(u)}, v{FWD(v)} {}

    template <int... O>
    auto operator()(dir_bounds<O>... bnds)
    {
        auto u_it = u(bnds...);
        auto v_it = v(bnds...);

        constexpr int N = sizeof...(O); // dims
        thrust::device_ptr<int> o = v.o;
        int n = std::get<N>(u.dir_bounds()).size();

        return thrust::make_transform_iterator(thrust::make_zip_iterator(u_it, v_it),
                                               dot_impl<N>{v_it, o, n});
    }
};

template <typename U, typename V>
dot_helper<U, V> make_dot_transform(U&& u, V&& v)
{
    return {FWD(u), FWD(v)};
}

//
//
//
template <int Shift, int N, typename Vec>
stencil_helper<Shift, Vec, N> make_stencil_transform(Vec&& vec)
{
    return {FWD(vec)};
}

//
// For some stencils we just need a shift by an offset rather than a full
// stencil/gradient.  In these cases we need to shift both the first and last indices by
// the specified amount
//
template <typename T, int I>
struct shift_helper : iter_math<shift_helper<T, I>> {
    T t;
    int s;

    shift_helper(T&& t, int s) : t{FWD(t)}, s{s} {}

    template <int... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t(bnds.shift(I == O ? s : 0)...);
    }
};

template <int N, typename Vec>
shift_helper<Vec, N> make_shift_transform(Vec&& vec, int s)
{
    return {FWD(vec), s};
}

template <typename T>
struct coarse_helper : iter_math<coarse_helper<T>> {
    T t;
    int r;

    coarse_helper(T&& t) : t{FWD(t)} {}

    template <int... F>
    constexpr auto operator()(dir_bounds<F>... fbnds)
    {
        return t.coarse_to_fine(fbnds...);
    }
};

template <typename T, int... Order, int... O>
auto make_coarse_transform(md_device_span<T, Order...>& v,
                           int ratio,
                           dir_bounds<O>... coarse_bnds)
{
}

//
// offset_helper's call operator will produce a
// matrix_traversal_iterator.  It is the responsibility of whatever is
// driving to form an offset iterator by saving the offset `o` and
// creating an offset_iterator at each deref.  This seems wasteful
//
template <typename T>
struct offset_helper : iter_math<offset_helper<T>> {
    T t;
    thrust::device_ptr<int> o;

    offset_helper(T&& t, thrust::device_ptr<int> o) : t{FWD(t)}, o{o} {}

    template <int... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t(std::bool_constant<true>{}, bnds...);
    }
};

template <typename Vec>
offset_helper<Vec> make_offset_transform(Vec&& v, thrust::device_ptr<int> o)
{
    return {FWD(v), o};
}

} // namespace lazy

//
// Facilitate an intuitive mathy semantics
// with_domain(k, j, i)(x = y + z)
//
// with_domain returns lambda which will invoke its arguments (assign_proxys) with its
// bounds
//
template <typename... P, typename = std::enable_if_t<(is_dir_bounds_v<P> && ...)>>
constexpr auto with_domain(P&&... ps)
{
    // should make a case for when no bounds are passed in.  In that case we should
    // return a lambda that will extract the bounds from the lhs of the assign proxy and
    // then call the proxy with the bounds
    if constexpr (sizeof...(ps) == 0)
        return [](auto&& assign_proxys) {
            // get the bounds from the lhs of the assignment expression
            auto tp = assign_proxys.u.dir_bounds();
            std::apply(FWD(assign_proxys), tp);
        };
    else
        return [=](auto&&... assign_proxys) { (assign_proxys(ps...), ...); };
}

template <typename W,
          typename... P,
          typename = std::enable_if_t<!is_dir_bounds_v<W> && (is_dir_bounds_v<P> && ...)>>
constexpr auto with_domain(W&& w, P&&... ps)
{

    return [=](auto&&... st_assign) mutable {
        // this is fine if all the st_assigns have numeric T's
        auto sz = (ps.size() * ...);
        if constexpr ((is_rhs_number_v<decltype(st_assign)> && ...)) {
            thrust::for_each_n(w(ps...), sz, lazy::tp_invoke{FWD(st_assign)...});
        } else if constexpr ((is_self_assign_proxy_v<decltype(st_assign)> && ...)) {
            thrust::for_each_n(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::make_zip_iterator(
                        thrust::make_tuple(FWD(st_assign).get_lhs_iterator(ps...)...)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        w(ps...), FWD(st_assign).get_iterator(ps...)...)))),

                sz,
                lazy::tp_layered_invoke{FWD(st_assign)...});
        } else {
            thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(
                                   w(ps...), FWD(st_assign).get_iterator(ps...)...)),
                               sz,
                               lazy::tp_invoke{FWD(st_assign)...});
        }
    };
}

//
// Facilitate an intuitive mathy semantics
// with_lhs_domain(x = y + z)
//
// Use the bounds associated with the lhs (x in this case) to evaluate the
// assign_proxy
//
template <typename T, typename = std::enable_if_t<is_assign_proxy_v<T>>>
constexpr auto with_lhs_domain(T&& t)
{
    return with_domain()(FWD(t));
}

//
// md_device_span class -> The primary interaction with be: construction, call operator
// (via with_domain), grad_x/y/z methods and math operators.  The dimension ordering is
// recorded via the Order parameter pack and used to faciliate seamless reording in the
// call operator
//
template <typename T, int... Order>
class md_device_span : lazy::iter_math<md_device_span<T, Order...>>
{

public:
    static constexpr auto N = sizeof...(Order);

    md_device_span() = default;

    md_device_span(T* v, lazy::dir_bounds<Order>... bnds) : v(v), b{bnds...} {}

    template <typename V>
    auto operator=(V&& v) &
    {
        return lazy::assign_proxy<md_device_span&, V>{*this, FWD(v)};
    }

    template <typename V, typename = std::enable_if_t<is_stencil_proxy_v<V>>>
    auto operator-=(V&& v) &
    {
        return lazy::self_assign_proxy<md_device_span&, decltype(-FWD(v) + *this)>{
            *this, -FWD(v) + *this};
    }

    // #define LAZY_VEC_OPERATORS(op, infix)                                                    //
//     template <typename V, typename = std::enable_if_t<is_stencil_proxy_v<V>>>            //
//     auto op(V&& v)&                                                                      //
//     {                                                                                    //
//         return lazy::self_assign_proxy<md_device_span&, decltype(*this infix FWD(v))>{      //
//             *this, *this infix FWD(v)};                                                  //
//     }

    //     LAZY_VEC_OPERATORS(operator+=, +)
    //     LAZY_VEC_OPERATORS(operator-=, -)
    //     LAZY_VEC_OPERATORS(operator*=, *)
    //     LAZY_VEC_OPERATORS(operator/=, /)
    // #undef LAZY_VEC_OPERATORS

    template <auto... O>
    auto operator()(lazy::dir_bounds<O>... bnds)
    {
        return call_helper<false>(begin(), b, bnds...);
    }

    template <auto... O>
    auto operator()(lazy::dir_bounds<O>... bnds) const
    {
        static_assert(N == sizeof...(O));
        return call_helper<false>(begin(), b, bnds...);
    }

    template <bool B, auto... O>
    auto operator()(std::bool_constant<B>, lazy::dir_bounds<O>... bnds)
    {
        return call_helper<B>(begin(), b, bnds...);
    }

    template <bool B, auto... O>
    auto operator()(std::bool_constant<B>, lazy::dir_bounds<O>... bnds) const
    {
        static_assert(N == sizeof...(O));
        return call_helper<B>(begin(), b, bnds...);
    }

    // call operator taking an int needs to return a function which takes the windowed
    // iterator and returns a reference to the thing
    auto operator()(int i) { return lazy::stencil_proxy<0>{i}; }
    auto operator()(int i1, int i2) { return lazy::stencil_proxy<1>{i1, i2}; }

    template <typename... Ints, typename = std::enable_if_t<(is_number_v<Ints> && ...)>>
    decltype(auto) at(Ints... is)
    {
        static_assert(sizeof...(Ints) == N);
        int c[] = {is...};
        int sz[N];

        for (int i = 0; i < N; i++) {
            c[i] -= b[i].lb();
            sz[i] = b[i].size();
        }
        return *(begin() + ravel<N>(sz, c));
    }

    template <int S = 1>
    auto grad_x(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::I>(*this, h);
    }

    template <int S = 1>
    auto stencil_x(mp::mp_int<S> = {})
    {
        return lazy::make_stencil_transform<S, lazy::dim::I>(*this);
    }

    auto shift_x(int s = 1) { return lazy::make_shift_transform<lazy::dim::I>(*this, s); }

    template <int S = 1>
    auto grad_y(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::J>(*this, h);
    }

    template <int S = 1>
    auto stencil_y(mp::mp_int<S> = {})
    {
        return lazy::make_stencil_transform<S, lazy::dim::J>(*this);
    }

    auto shift_y(int s = 1) { return lazy::make_shift_transform<lazy::dim::J>(*this, s); }

    template <int S = 1>
    auto grad_z(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::K>(*this, h);
    }

    template <int S = 1>
    auto stencil_z(mp::mp_int<S> = {})
    {
        return lazy::make_stencil_transform<S, lazy::dim::K>(*this);
    }

    auto shift_z(int s = 1) { return lazy::make_shift_transform<lazy::dim::K>(*this, s); }

    // corner shifts
    auto shift_xy(int xs = 1, int ys = 1)
    {
        return lazy::make_shift_transform<lazy::dim::J>(shift_x(xs), ys);
    }

    auto shift_xz(int xs = 1, int zs = 1)
    {
        return lazy::make_shift_transform<lazy::dim::K>(shift_x(xs), zs);
    }

    auto shift_yz(int ys = 1, int zs = 1)
    {
        return lazy::make_shift_transform<lazy::dim::K>(shift_y(ys), zs);
    }

    auto offset(thrust::device_ptr<int> o)
    {
        return lazy::make_offset_transform(*this, o);
    }

    template <typename U>
    auto dot(U&& u)
    {
        // requires last(Order...) == W
        return lazy::make_dot_transform(*this, FWD(u));
    }

    __host__ __device__ auto size() const { return v.size(); }

    //
    // Window iterator infrastructure
    //
    auto window()
    {
        return [this](auto&&... bnds) mutable { return (*this)(FWD(bnds)...); };
    }

    //
    // coarse_to_fine
    //
    template <auto... O>
    auto fine(int r, lazy::dir_bounds<O>... bnds)
    {
        // assert((bnds.size() == 1) && ...);
        return [this, r, bnds...](auto&&... fbnds) mutable {
            return coarse_to_fine(r, std::tuple{bnds...}, fbnds...);
        };
    }

    template <auto... CO, auto... FO>
    auto coarse_to_fine(int r,
                        std::tuple<lazy::dir_bounds<CO>...> cbnds,
                        lazy::dir_bounds<FO>... fbnds)
    {
        static_assert(sizeof...(CO) + 1 == sizeof...(FO));

        // Determine the dimension we are iterating over:
        using Coarse = index_list<CO...>;
        using Fine = index_list<FO...>;
        using Base = index_list<Order...>;
        static constexpr auto I = missing_index_v<Fine, Coarse>;

        // Extract the fine index lower bound
        std::tuple<lazy::dir_bounds<FO>&...> f{fbnds...};
        int fi = std::get<map_index_v<Fine, I>>(f).lb();

        // Put coarse lower bounds in Order....
        int lb_bnds[N] = {std::get<map_index_v<Coarse, CO>>(cbnds).lb()...};
        int lb[] = {lb_bnds[map_index_v<Coarse, Order>]...};

        // Insert baseline coarse index and make them zero based
        lb[map_index_v<Base, I>] = detail::coarse_index(r, fi);

        int sz[N];

        for (int i = 0; i < N; i++) {
            lb[i] -= b[i].lb();
            sz[i] = b[i].size();
        }

        return make_coarse_to_fine<map_index_v<Base, I>, N>(begin(), sz, lb, fi, r);
    }

    auto begin() { return v; }
    auto begin() const { return v; }
    auto end() { return v; }
    auto end() const { return v; }

    auto copy_to(T*) const
    { /* return thrust::copy(begin(), end(), t); */
    }

    std::tuple<lazy::dir_bounds<Order>...> dir_bounds() const
    {
        return {b[map_index_v<index_list<Order...>, Order>]...};
    }

private:
    template <bool iter_flag, typename Iter, auto... O>
    static auto call_helper(Iter it, const bounds (&b)[N], lazy::dir_bounds<O>... bnds)
    {
        // When the input bounds order `O` is different from the initialized order
        // `Order`, we return a submatrix transpose iterator.  The first step is to put
        // all the bounds data in the intialization order and then use the input bounds
        // order `O` to form the transpose

        int lb_bnds[N] = {bnds.lb()...};
        int ub_bnds[N] = {bnds.ub()...};
        int stm_bnds[N] = {bnds.stride...};

        int lb[] = {lb_bnds[map_index_v<index_list<O...>, Order>]...};
        int ub[] = {ub_bnds[map_index_v<index_list<O...>, Order>]...};
        int stm[N] = {stm_bnds[map_index_v<index_list<O...>, Order>]...};

        int sz[N];

        // If N > sizeof...(O) we are in creating a window iterator over the specified
        // domain.  In this situation, the last entry in lb/ub is zero and should be
        // corrected to the window bounds to ensure proper creation in the
        // submatrix_helper
        for (int i = sizeof...(O); i < N; i++) {
            lb[i] = b[i].lb();
            ub[i] = b[i].ub();
            stm[i] = b[i].stride;
        }

        for (int i = 0; i < N; i++) {
            lb[i] -= b[i].lb();
            ub[i] -= b[i].lb();
            sz[i] = b[i].size();
        }

        using Seq = transpose_sequence_t<index_list<Order...>, index_list<O...>>;
        return detail::make_submatrix_helper<N>(
            Seq{}, it, sz, lb, ub, stm, std::bool_constant<iter_flag>{});
    }

    thrust::device_ptr<T> v;
    bounds b[N];
};

template <typename T, auto... Order>
md_device_span<T, Order...> make_md_span(T* t, const lazy::dir_bounds<Order>&... bnds)
{
    return {t, bnds.unit_stride()...};
}

template <typename T, auto... Order>
md_device_span<const T, Order...> make_md_span(const T* t,
                                               const lazy::dir_bounds<Order>&... bnds)
{
    return {t, bnds.unit_stride()...};
}

template <typename T, auto... Order>
md_device_span<T, Order...>
make_md_span(T* t, int offset, const lazy::dir_bounds<Order>&... bnds)
{

    // return {t, bnds.expand(offset)...};
    return {t, detail::expand_bounds(offset, bnds.unit_stride())...};
}

template <typename T, auto... Order>
md_device_span<const T, Order...>
make_md_span(const T* t, int offset, const lazy::dir_bounds<Order>&... bnds)
{

    // return {t, bnds.expand(offset)...};
    return {t, detail::expand_bounds(offset, bnds.unit_stride())...};
}

template <typename T>
md_device_span<T, lazy::dim::W, lazy::dim::I> make_offset_span(T* t, int n, int dims)
{
    return {t, Wb{1, n}, Ib{1, dims}};
}

template <typename T>
md_device_span<const T, lazy::dim::W, lazy::dim::I>
make_offset_span(const T* t, int n, int dims)
{
    return {t, Wb{0, n - 1}, Ib{0, dims - 1}};
}
