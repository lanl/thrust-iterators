#pragma once

#include "lazy_math.hpp"
#include "md_device_vector.hpp"
#include "traits.hpp"

#include <thrust/execution_policy.h>
#include <type_traits>

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
};

namespace dim
{
enum { K = 0, J = 1, I = 2 };
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
struct gradient_helper {
    T t;

    template <auto... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t((bnds + shift_v<Shift, I, O>)...)
            .template stencil<map_index_v<index_list<O...>, I>>();
    }
};

template <int Shift, auto N, typename Vec, typename T>
transform_op<T, gradient_helper<Shift, Vec, N>, gradient>
make_gradient_transform(Vec&& vec, T h)
{
    return {h, {FWD(vec)}};
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

//
// Facilitate an intuitive mathy semantics
// with_lhs_domain(x = y + z)
//
// Use the bounds associated with the lhs (x in this case) to evaluate the assign_proxy
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

    template <auto... O>
    auto operator()(lazy::dir_bounds<O>... bnds)
    {
        static_assert(N == sizeof...(O));

        // When the input bounds order `O` is different from the initialized order
        // `Order`, we return a submatrix transpose iterator.  The first step is to put
        // all the bounds data in the intialization order and then use the input bounds
        // order `O` to form the transpose

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

        if constexpr (((O == Order) && ...)) {
            return make_submatrix(begin(), sz, lb, ub);
        } else {
            using Seq = transpose_sequence_t<index_list<Order...>, index_list<O...>>;
            return detail::make_submatrix_helper<decltype(begin()), N>(
                Seq{}, begin(), sz, lb, ub);
        }
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

        if constexpr (((O == Order) && ...)) {
            return make_submatrix(begin(), sz, lb, ub);
        } else {
            using Seq = transpose_sequence_t<index_list<Order...>, index_list<O...>>;
            return detail::make_submatrix_helper(Seq{}, begin(), sz, lb, ub);
        }
    }

    template <int S = 1>
    auto grad_x(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::I>(*this, h);
    }

    template <int S = 1>
    auto grad_y(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::J>(*this, h);
    }

    template <int S = 1>
    auto grad_z(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S, lazy::dim::K>(*this, h);
    }

    __host__ __device__ auto size() const { return v.size(); }

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
lazy_vector<T, Order...> make_vec(const T* t, const lazy::dir_bounds<Order>... bnds)
{
    return {t, bnds...};
}

template <typename T, auto... Order>
lazy_vector<T, Order...>
make_vec(const T* t, int offset, const lazy::dir_bounds<Order>... bnds)
{

    return {t, bnds.expand(offset)...};
}
