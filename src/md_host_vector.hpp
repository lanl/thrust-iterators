#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <vector>

struct hbounds {
    int first;
    int last;

    hbounds() = default;
    hbounds(int first, int last, bool inclusive = true)
        : first{first}, last{last + inclusive}
    {
    }

    int size() const { return last - first; }
};

template <typename T, int Dim>
class md_host_vector
{
public:
    md_host_vector() = default;
    md_host_vector(std::array<hbounds, Dim> b)
        : v(std::transform_reduce(b.begin(),
                                  b.end(),
                                  1,
                                  std::multiplies{},
                                  [](auto&& b) { return b.size(); })),
          b{b}
    {
    }

    template <typename... Hbounds>
    md_host_vector(hbounds b0, Hbounds&&... bnds)
        : md_host_vector(std::array{b0, std::forward<Hbounds>(bnds)...})
    {
        static_assert(Dim == 1 + sizeof...(Hbounds));
    }

    const T* data() const { return v.data(); }
    T* data() { return &v[0]; }

    const auto& vec() const { return v; }
    auto begin() { return v.begin(); }
    auto begin() const { return v.begin(); }
    auto end() { return v.end(); }
    auto end() const { return v.end(); }

    auto size() const { return v.size(); }

private:
    std::vector<T> v;
    std::array<hbounds, Dim> b;
};
