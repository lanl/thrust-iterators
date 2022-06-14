struct bounds {
    int first;
    int last;

    bounds() = default;
    constexpr bounds(int first, int last, bool inclusive = true)
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
template <int I>
struct dir_bounds : bounds {
    dir_bounds() = default;
    constexpr dir_bounds(int f, int l, bool inclusive = true) : bounds(f, l, inclusive) {}
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