#pragma once

#include "sliding_iterator.hpp"
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

    auto begin() { return v.begin(); }
    auto begin() const { return v.begin(); }
    auto end() { return v.end(); }
    auto end() const { return v.end(); }

private:
    thrust::device_vector<T> v;
    std::array<bounds, Dim> b;
};

template <typename T, typename... Bounds>
md_vector<T, sizeof...(Bounds)> make_md_vec(const T* t, Bounds&&... bnds)
{
    return {t, std::array{std::forward<Bounds>(bnds)...}};
}
