#pragma once

#include "lazy_math.hpp"
#include "stencil_proxy.hpp"
#include "thrust/for_each.h"
#include "traits.hpp"

#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility>

#include "submatrix_iterator.hpp"

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

struct bounds {
    int first;
    int last;

    bounds() = default;
    bounds(int first, int last, bool inclusive = true)
        : first{first}, last{last + inclusive}
    {
    }

    // operator+ increments the last bound
    bounds& operator+=(int x)
    {
        last += x;
        return *this;
    }
    bounds friend operator+(bounds b, int x)
    {
        b += x;
        return b;
    }
    // operator- decrements the first bound
    bounds& operator-=(int x)
    {
        first -= x;
        return *this;
    }

    bounds friend operator-(bounds b, int x)
    {
        b -= x;
        return b;
    }

    bounds expand(int x) const
    {
        bounds b{*this};
        b -= x;
        b += x;
        return b;
    }

    int lb() const { return first; }
    int ub() const { return last - 1; }

    int size() const { return last - first; }
};

// Generally, the user will not be required to use anything in the `lazy` namespace
namespace lazy
{

// The data coming from amp are multidimensional fortran arrays.  Using `bounds` allows
// for intuitive construction of our lazy vectors.  Using the templated `dir_bounds`
// allows us to record the order of the data.  cell/node data are standard KJI order, face
// data are all different.  Recordning the constructed order faciliates automatic
// transpose iterators in the lazy_vec call operator
//
// The bounds on the incoming data are generally perturbations around some "base".  We use
// +/-/expand to express that perturbation
template <auto I>
struct dir_bounds : bounds {
    dir_bounds() = default;
    dir_bounds(int f, int l, bool inclusive = true) : bounds(f, l, inclusive) {}
    dir_bounds(const bounds& bnd) : bounds(bnd) {}

    dir_bounds friend operator+(dir_bounds b, int x)
    {
        // ensure that adding a negative number adjust the lower bound
        if (x >= 0)
            b += x;
        else
            b -= (-x);

        return b;
    }

    dir_bounds friend operator-(dir_bounds b, int x)
    {
        b -= x;
        return b;
    }

    dir_bounds expand(int x) const
    {
        dir_bounds b{*this};
        b -= x;
        b += x;
        return b;
    }

    dir_bounds shift(int x) const
    {
        dir_bounds b{*this};
        b.first += x;
        b.last += x;
        return b;
    }
};

namespace dim
{
enum { K = 0, J, I, W };
}
} // namespace lazy

//
// These are the types the user will use to construct bounds in the I, J, and K
// directions.
//
using Ib = lazy::dir_bounds<lazy::dim::I>;
using Jb = lazy::dir_bounds<lazy::dim::J>;
using Kb = lazy::dir_bounds<lazy::dim::K>;

//
// A separate type for "window" bounds needed for the window iterator
//
using Wb = lazy::dir_bounds<lazy::dim::W>;

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
template <int Shift, typename T, auto I>
struct stencil_helper : lazy_vec_math<stencil_helper<Shift, T, I>> {
    T t;

    stencil_helper(T&& t) : t{FWD(t)} {}

    template <auto... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t((bnds + shift_v<Shift, I, O>)...)
            .template stencil<map_index_v<index_list<O...>, I>>();
    }
};

template <int Shift, auto N, typename Vec, typename T>
transform_op<T, stencil_helper<Shift, Vec, N>, gradient>
make_gradient_transform(Vec&& vec, T h)
{
    return {h, {FWD(vec)}};
}

//
//
//
template <int Shift, auto N, typename Vec>
stencil_helper<Shift, Vec, N> make_stencil_transform(Vec&& vec)
{
    return {FWD(vec)};
}

//
// For some stencils we just need a shift by an offset rather than a full
// stencil/gradient.  In these cases we need to shift both the first and last indices by
// the specified amount
//
template <typename T, auto I>
struct shift_helper : lazy_vec_math<shift_helper<T, I>> {
    T t;
    int s;

    shift_helper(T&& t, int s) : t{FWD(t)}, s{s} {}

    template <auto... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t(bnds.shift(I == O ? s : 0)...);
    }
};

template <auto N, typename Vec>
shift_helper<Vec, N> make_shift_transform(Vec&& vec, int s)
{
    return {FWD(vec), s};
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
// lazy_vector class -> The primary interaction with be: construction, call operator (via
// with_domain), grad_x/y/z methods and math operators.  The dimension ordering is
// recorded via the Order parameter pack and used to faciliate seamless reording in the
// call operator
//
template <typename T, auto... Order>
class lazy_vector : lazy::lazy_vec_math<lazy_vector<T, Order...>>
{

public:
    static constexpr auto N = sizeof...(Order);

    lazy_vector() = default;

    lazy_vector(const T* v, lazy::dir_bounds<Order>... bnds)
        : v(v, v + (bnds.size() * ...)), b{bnds...}
    {
    }

    template <typename V>
    auto operator=(V&& v) &
    {
        return lazy::assign_proxy<lazy_vector&, V>{*this, FWD(v)};
    }

    template <typename V, typename = std::enable_if_t<is_stencil_proxy_v<V>>>
    auto operator-=(V&& v) &
    {
        return lazy::self_assign_proxy<lazy_vector&, decltype(-FWD(v) + *this)>{
            *this, -FWD(v) + *this};
    }

    // #define LAZY_VEC_OPERATORS(op, infix)                                                    \
//     template <typename V, typename = std::enable_if_t<is_stencil_proxy_v<V>>>            \
//     auto op(V&& v)&                                                                      \
//     {                                                                                    \
//         return lazy::self_assign_proxy<lazy_vector&, decltype(*this infix FWD(v))>{      \
//             *this, *this infix FWD(v)};                                                  \
//     }

    //     LAZY_VEC_OPERATORS(operator+=, +)
    //     LAZY_VEC_OPERATORS(operator-=, -)
    //     LAZY_VEC_OPERATORS(operator*=, *)
    //     LAZY_VEC_OPERATORS(operator/=, /)
    // #undef LAZY_VEC_OPERATORS

    template <auto... O>
    auto operator()(lazy::dir_bounds<O>... bnds)
    {
        // static_assert(N == sizeof...(O));

        // When the input bounds order `O` is different from the initialized order
        // `Order`, we return a submatrix transpose iterator.  The first step is to put
        // all the bounds data in the intialization order and then use the input bounds
        // order `O` to form the transpose

        int lb_bnds[N] = {bnds.lb()...};
        int ub_bnds[N] = {bnds.ub()...};
        int lb[] = {lb_bnds[map_index_v<index_list<O...>, Order>]...};
        int ub[] = {ub_bnds[map_index_v<index_list<O...>, Order>]...};

        int sz[N];

        // If N > sizeof...(O) we are in creating a window iterator over the specified
        // domain.  In this situation, the last entry in lb/ub is zero and should be
        // corrected to the window bounds to ensure proper creation in the
        // submatrix_helper
        for (int i = sizeof...(O); i < N; i++) {
            lb[i] = b[i].lb();
            ub[i] = b[i].ub();
        }

        for (int i = 0; i < N; i++) {
            lb[i] -= b[i].lb();
            ub[i] -= b[i].lb();
            sz[i] = b[i].size();
        }

        using Seq = transpose_sequence_t<index_list<Order...>, index_list<O...>>;
        return detail::make_submatrix_helper<N>(Seq{}, begin(), sz, lb, ub);
    }

    template <auto... O>
    auto operator()(lazy::dir_bounds<O>... bnds) const
    {
        static_assert(N == sizeof...(O));

        int lb_bnds[] = {bnds.lb()...};
        int ub_bnds[] = {bnds.ub()...};
        int lb[] = {lb_bnds[map_index_v<index_list<O...>, Order>]...};
        int ub[] = {ub_bnds[map_index_v<index_list<O...>, Order>]...};

        int sz[N];

        for (int i = 0; i < N; i++) {
            lb[i] -= b[i].lb();
            ub[i] -= b[i].lb();
            sz[i] = b[i].size();
        }

        using Seq = transpose_sequence_t<index_list<Order...>, index_list<O...>>;
        return detail::make_submatrix_helper<N>(Seq{}, begin(), sz, lb, ub);
    }

    // call operator taking an int needs to return a function which takes the windowed
    // iterator and returns a reference to the thing
    auto operator()(int i) { return lazy::stencil_proxy<0>{i}; }
    auto operator()(int i1, int i2) { return lazy::stencil_proxy<2>{i1, i2}; }

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

    __host__ __device__ auto size() const { return v.size(); }

    //
    // Window iterator infrastructure
    //
    auto window()
    {
        return [this](auto&&... bnds) mutable { return (*this)(FWD(bnds)...); };
    }

    auto begin() { return v.begin(); }
    auto begin() const { return v.begin(); }
    auto end() { return v.end(); }
    auto end() const { return v.end(); }

    auto copy_to(T* t) const { return thrust::copy(begin(), end(), t); }

    std::tuple<lazy::dir_bounds<Order>...> dir_bounds() const
    {
        return {b[map_index_v<index_list<Order...>, Order>]...};
    }

private:
    thrust::device_vector<T> v;
    bounds b[N]; // std::array<bounds, N> b;
};

template <typename T, auto... Order>
lazy_vector<T, Order...> make_vec(const T* t, const lazy::dir_bounds<Order>&... bnds)
{
    return {t, bnds...};
}

namespace detail
{
template <auto N>
constexpr auto expand_bounds(int offset, const lazy::dir_bounds<N>& b)
{
    if constexpr (N == lazy::dim::W)
        return b;
    else
        return b.expand(offset);
}
} // namespace detail

template <typename T, auto... Order>
lazy_vector<T, Order...>
make_vec(const T* t, int offset, const lazy::dir_bounds<Order>&... bnds)
{

    // return {t, bnds.expand(offset)...};
    return {t, detail::expand_bounds(offset, bnds)...};
}
