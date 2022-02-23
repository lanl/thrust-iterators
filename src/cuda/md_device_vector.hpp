#pragma once

#include "sliding_iterator.hpp"
#include "submatrix_iterator.hpp"
#include "window_iterator.hpp"
#include <numeric>
#include <thrust/device_vector.h>

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

template <typename Iter>
struct offset_pair {
    Iter first;
    Iter last;
};

namespace detail
{
template <auto Dim, auto Dim2>
int col_offset(const std::array<bounds, Dim>& b, const std::array<int, Dim2>& c)
{
    static_assert(Dim >= Dim2);

    int offset = 0;
    for (int i = 0; i < Dim - 1; i++)
        offset = (offset + (c[i] - b[i].first)) * b[i + 1].size();

    if constexpr (Dim == Dim2) {
        return offset + (c[Dim - 1] - b[Dim - 1].first);
    } else
        return offset;
}
} // namespace detail

template <typename T, int Dim>
class md_vector
{
public:
    md_vector() = default;
    md_vector(const T* v, std::array<bounds, Dim> b)
        : v(v,
            v + std::accumulate(b.begin(),
                                b.end(),
                                1,
                                [](auto&& acc, auto&& x) { return acc * x.size(); })),
          b{b}
    {
    }

    template <typename... Bounds>
    md_vector(const T* v, bounds b0, Bounds&&... bnds)
        : md_vector(v, std::array{b0, std::forward<Bounds>(bnds)...})
    {
        static_assert(Dim == 1 + sizeof...(Bounds));
    }

    // returns a window iterator with a window size equal to the last dimension
    auto column(std::array<int, Dim - 1> coords)
    {
        int offset = detail::col_offset(b, coords);
        return make_window(v.begin() + offset, b.back().size());
    }

    auto column(std::array<int, Dim - 1> coords) const
    {
        int offset = detail::col_offset(b, coords);
        return make_window(v.cbegin() + offset, b.back().size());
    }

    // for column range [first, last], keeping the last as inclusive here since we are
    // mapping fortran kernels
    window_pair<typename thrust::device_vector<T>::iterator>
    column(std::array<int, Dim - 1> c0, std::array<int, Dim - 1> c1)
    {
        int window_size = b.back().size();
        return {make_window(v.begin() + detail::col_offset(b, c0), window_size),
                make_window(v.begin() + detail::col_offset(b, c1) + window_size,
                            window_size)};
    }

    window_pair<typename thrust::device_vector<T>::const_iterator>
    column(std::array<int, Dim - 1> c0, std::array<int, Dim - 1> c1) const
    {
        int window_size = b.back().size();
        return {make_window(v.cbegin() + detail::col_offset(b, c0), window_size),
                make_window(v.cbegin() + detail::col_offset(b, c1) + window_size,
                            window_size)};
    }

    auto sliding(int window_size, std::array<int, Dim> c)
    {
        return make_sliding(v.begin() + detail::col_offset(b, c), window_size);
    }

    auto sliding(int window_size, std::array<int, Dim> c) const
    {
        return make_sliding(v.cbegin() + detail::col_offset(b, c), window_size);
    }

    sliding_pair<typename thrust::device_vector<T>::iterator>
    sliding(int window_size, std::array<int, Dim> c0, std::array<int, Dim> c1)
    {
        return {make_sliding(v.begin() + detail::col_offset(b, c0), window_size),
                make_sliding(v.begin() + detail::col_offset(b, c1) + window_size - 1,
                             window_size)};
    }

    sliding_pair<typename thrust::device_vector<T>::const_iterator>
    sliding(int window_size, std::array<int, Dim> c0, std::array<int, Dim> c1) const
    {
        return {make_sliding(v.cbegin() + detail::col_offset(b, c0), window_size),
                make_sliding(v.cbegin() + detail::col_offset(b, c1) + window_size - 1,
                             window_size)};
    }

    // return the underlying vector iterator properly offset
    offset_pair<typename thrust::device_vector<T>::iterator>
    offset(std::array<int, Dim> c0, std::array<int, Dim> c1)
    {
        return {v.begin() + detail::col_offset(b, c0),
                v.begin() + detail::col_offset(b, c1) + 1};
    }

    offset_pair<typename thrust::device_vector<T>::const_iterator>
    offset(std::array<int, Dim> c0, std::array<int, Dim> c1) const
    {
        return {v.begin() + detail::col_offset(b, c0),
                v.begin() + detail::col_offset(b, c1) + 1};
    }

    template <typename... Bounds>
    auto operator()(Bounds&&... bnds)
    {
        static_assert(Dim == sizeof...(Bounds));

        int lb[] = {bnds.lb()...};
        int ub[] = {bnds.ub()...};
        int sz[Dim];

        for (int i = 0; i < Dim; i++) {
            lb[i] -= b[i].lb();
            ub[i] -= b[i].lb();
            sz[i] = b[i].size();
        }

        return make_submatrix(begin(), sz, lb, ub);
    }

    auto stencil(int I)
    {
        int dims[Dim];
        int n[Dim];
        for (int i = 0; i < Dim; i++) {
            dims[i] = i == I ? b[i].size() - 1 : b[i].size();
            n[i] = b[i].size();
        }

        auto stride = ::stride_dim<Dim>(n, I);
        auto limit = ::stride_dim<Dim>(dims, I - 1);
        auto sz = ::stride_dim<Dim>(dims, -1);

        return make_forward_stencil(begin(), stride, limit, sz);
    }

    // The baseline format is c-ordering of 0, 1, 2
    template <auto... I>
    auto transpose() const
    {
        int n[Dim];
        for (int i = 0; i < Dim; i++) n[i] = b[i].size();
        return make_transpose<I...>(begin(), n);
    }

    auto istencil() { return stencil(Dim - 1); }
    auto jstencil() { return stencil(Dim - 2); }
    auto kstencil() { return stencil(Dim - 3); }

    // permutes from 0, 1 -> 1, 0
    auto ij() const { return this->transpose<1, 0>(); }
    auto ikj() const { return this->transpose<2, 0, 1>(); }
    auto jik() const { return this->transpose<1, 2, 0>(); }

    auto size() const { return v.size(); }

    auto begin() { return v.begin(); }
    auto begin() const { return v.begin(); }
    auto end() { return v.end(); }
    auto end() const { return v.end(); }

private:
    thrust::device_vector<T> v;
    std::array<bounds, Dim> b;
};

template <typename T, typename... Bounds>
md_vector<T, 1 + sizeof...(Bounds)>
make_md_vec(const T* t, const bounds& b, Bounds&&... bnds)
{
    return {t, std::array{b, std::forward<Bounds>(bnds)...}};
}

template <typename T, typename... Bounds>
md_vector<T, 1 + sizeof...(Bounds)>
make_md_vec(const T* t, int offset, bounds b, Bounds... bnds)
{

    return {t, std::array{b.expand(offset), bnds.expand(offset)...}};
}
