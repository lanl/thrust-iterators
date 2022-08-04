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


#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda/matrix_utils.hpp"
#include "cuda/md_bounds.hpp"

#include "random.hpp"

// Container to ensure memory is allocated on device since the kernels
// assume all data movement is device->device and do no allocation or extra
// copying.

template <typename T, int... Order>
class md_device_vector
{

public:
    static constexpr auto N = sizeof...(Order);

    md_device_vector() = default;
    md_device_vector(lazy::dir_bounds<Order>... bnds)
        : h(detail::bounds_sz(bnds...)), d(h), b{bnds...}, sync{true}
    {
    }

    const T* data() const
    {
        // assume that the purpose of getting a point is to change the data
        sync = false;
        return thrust::raw_pointer_cast(d.data());
    }

    T* data()
    {
        sync = false;
        return thrust::raw_pointer_cast(d.data());
    }

    const T* host_data() const
    {
        // assume that the purpose of getting a point is to change the data
        // sync = false;
        return h.data();
    }

    T* host_data()
    {
        // sync = false;
        return h.data();
    }

    const thrust::host_vector<T>& host()
    {
        if (!sync) {
            sync = true;
            h = d;
        }
        return h;
    }

    operator std::vector<T>()
    {

        std::vector<T> u(h.size());
        if (!sync) {
            sync = true;
            // assume device has correct info
            h = d;
        }

        thrust::copy(h.begin(), h.end(), u.begin());
        return u;
    }

    template <typename... Args>
    int index(Args... args) const
    {
        static_assert(sizeof...(Args) == N);
        int c[] = {args...};
        int sz[N];
        for (int i = 0; i < N; i++) {
            c[i] -= b[i].lb();
            sz[i] = b[i].size();
        }
        return ravel<N>(sz, c);
    }

    int size() const { return h.size(); }

    int dim(int i) const { return b[i].size(); }

    void fill_random(T x0 = 0, T x1 = 1)
    {
        auto f = [x0, x1]() { return pick(x0, x1); };
        std::generate(h.begin(), h.end(), f);
        d = h;
        sync = true;
    }

private:
    thrust::host_vector<T> h;
    thrust::device_vector<T> d;
    bounds b[N];
    mutable bool sync;
};

template <typename T, auto... Order>
md_device_vector<T, Order...> make_md_vec(const lazy::dir_bounds<Order>&... bnds)
{
    return {bnds...};
}

template <typename T, auto... Order>
md_device_vector<T, Order...> make_md_vec(int offset,
                                          const lazy::dir_bounds<Order>&... bnds)
{

    return {detail::expand_bounds(offset, bnds)...};
}

template <typename T>
std::vector<T> to_std(const thrust::device_vector<T>& a)
{
    std::vector<T> b(a.size());
    thrust::copy(a.begin(), a.end(), b.begin());
    return b;
}
