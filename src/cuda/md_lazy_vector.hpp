#pragma once

#include "lazy_math.hpp"
#include "md_device_vector.hpp"
#include "traits.hpp"

#include <thrust/execution_policy.h>
#include <type_traits>

// 1. make bounds templated so we dont have to repeat the order using placeholders
// DONE!

// 2. Hardcode transpose operators via specialization
// Transpose DONE! - No hardcoded operators needed

// 3. instead of stencil, define grad_x, grad_y, grad_z
// DONE!

// 4. apply multiple operations

// 5. combine gradients and such

static constexpr auto up = mp::mp_int<1>{};
static constexpr auto down = mp::mp_int<-1>{};

namespace lazy
{
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

template <auto I, typename T>
struct bound_dim {
    T t;
};

namespace placeholders
{

// enum dim { K = 0, J, I };

namespace detail
{
template <auto N>
struct placeholder_t {
    static constexpr auto value = N;

    template <typename T>
    bound_dim<N, T> constexpr operator=(T&& bnds) const
    {
        return {FWD(bnds)};
    }
};

} // namespace detail

static constexpr auto K = detail::placeholder_t<0>{};
static constexpr auto J = detail::placeholder_t<1>{};
static constexpr auto I = detail::placeholder_t<2>{};

} // namespace placeholders

template <typename U, typename V>
struct assign_proxy {
    U u;
    V v;

    template <typename... Bnds>
    void operator()(Bnds&&... bnds)
    {
        auto out = u(bnds...);
        thrust::copy_n(thrust::device, v(FWD(bnds)...), out.size(), out);
    }
};

template <int Shift, typename T, auto I>
struct gradient_helper {
    T t;

    template <auto... O>
    constexpr auto operator()(dir_bounds<O>... bnds)
    {
        return t((bnds + shift_v<Shift, I, O>)...)
            .template stencil<map_index_v<index_list<O...>, I>>();
    }

    template <typename... Dims,
              typename = std::enable_if_t<(is_bound_dim_v<Dims> && ...)>>
    auto operator()(Dims&&... dims)
    {
        // for now assume that the bound dims are in the same order
        return (*this)(FWD(dims).t...);
    }
};

template <int Shift, typename Vec, typename T, auto N>
transform_op<T, gradient_helper<Shift, Vec, N>, gradient>
make_gradient_transform(Vec&& vec, T h, placeholders::detail::placeholder_t<N>)
{
    return {h, {FWD(vec)}};
}
} // namespace lazy

template <typename... P, typename = std::enable_if_t<(is_bound_dim_v<P> && ...)>>
constexpr auto with_domain(P&&... ps)
{
    // should make a case for when no bound_dims are passed in.  In that case we should
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

template <typename T, typename = std::enable_if_t<is_assign_proxy_v<T>>>
constexpr auto with_lhs_domain(T&& t)
{
    return with_domain()(FWD(t));
}

using Ib = lazy::dir_bounds<2>;
using Jb = lazy::dir_bounds<1>;
using Kb = lazy::dir_bounds<0>;

template <typename T, auto... Order>
class lazy_vector : lazy::lazy_vec_math<lazy_vector<T, Order...>>
{
    friend class lazy::vec_math_access;

public:
    static constexpr auto N = sizeof...(Order);

    lazy_vector() = default;

    lazy_vector(const T* v, lazy::dir_bounds<Order>... bnds)
        : v(v, v + (bnds.size() * ...)), b{bnds...}
    {
    }

    template <typename V>
    auto operator=(V&& v)
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

    template <typename... Dims,
              typename = std::enable_if_t<(is_bound_dim_v<Dims> && ...)>>
    auto operator()(Dims&&... dims)
    {
        return (*this)(FWD(dims).t...);
    }

    template <typename... Dims,
              typename = std::enable_if_t<(is_bound_dim_v<Dims> && ...)>>
    auto operator()(Dims&&... dims) const
    {
        return (*this)(FWD(dims).t...);
    }

    template <int S = 1>
    auto grad_x(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S>(*this, h, lazy::placeholders::I);
    }

    template <int S = 1>
    auto grad_y(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S>(*this, h, lazy::placeholders::J);
    }

    template <int S = 1>
    auto grad_z(T h, mp::mp_int<S> = {})
    {
        return lazy::make_gradient_transform<S>(*this, h, lazy::placeholders::K);
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
