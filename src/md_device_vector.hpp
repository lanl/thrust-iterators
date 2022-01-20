#pragma once

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

private:
    thrust::device_vector<T> v;
    std::array<bounds, Dim> b;
};

template <typename T, typename... Bounds>
md_vector<T, sizeof...(Bounds)> make_md_vec(const T* t, Bounds&&... bnds)
{
    return {t, std::array{std::forward<Bounds>(bnds)...}};
}
