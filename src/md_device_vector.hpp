#pragma once

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
        //sync = false;
        return h.data();
    }

    T* host_data()
    {
        //sync = false;
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

    int size() const { return h.size(); }

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

template<typename T>
std::vector<T> to_std(const thrust::device_vector<T>& a) {
    std::vector<T> b(a.size());
    thrust::copy(a.begin(), a.end(), b.begin());
    return b;
}
