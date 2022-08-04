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


struct bounds {
    int first;
    int last;
    int stride;

    bounds() = default;
    constexpr bounds(int first, int last, int stride = 1, bool inclusive = true)
        : first{first}, last{last + inclusive}, stride{stride}
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

    int size() const { return (last - first) / stride; }
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
template <int I>
struct dir_bounds : bounds {
    dir_bounds() = default;
    constexpr dir_bounds(int f, int l, int stride = 1, bool inclusive = true)
        : bounds(f, l, stride, inclusive)
    {
    }
    constexpr dir_bounds(const bounds& bnd) : bounds(bnd) {}

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

    dir_bounds unit_stride() const
    {
        dir_bounds b{*this};
        b.stride = 1;
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

template <int... Order>
auto bounds_sz(const lazy::dir_bounds<Order>&... bnds)
{
    return (bnds.size() * ...);
}

} // namespace detail
