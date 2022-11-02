// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#include "../transpose_iterator_test.hpp"
#include "submatrix_iterator.hpp"
#include "transpose_iterator.hpp"

#include <cassert>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>

template <typename T, auto... I>
void transpose_test_cuda<T, I...>::init(const T* v, const int (&sz)[N], T* u)
{
    int z = sz[0];
    for (int i = 1; i < N; i++) z *= sz[i];

    thrust::device_vector<T> x(v, v + z);
    auto first = make_transpose<I...>(x.begin(), sz);
    auto last = first + z;
    assert(thrust::distance(first, last) == z);

    thrust::copy_n(first, z, u);

    // int z = sz[0];
    // for (int i = 1; i < N; i++) z *= sz[i];
    // int n[N];
    // for (int i = 0; i < N; i++) n[i] = ub[i] - lb[i] + 1;

    // thrust::device_vector<T> x(v, v + z);
    // printf("vector sz: %d//n", z);
    // {
    //     auto s = make_submatrix(x.begin(), sz, lb, ub);
    //     //        s.print_current();
    //     *s = -1;
    //     //        s.print_current();
    //     assert(s[0] == -1);
    //     auto ss = s + 1;

    //     assert(thrust::distance(s, ss) == 1);

    //     ++s;
    //     --s;
    //     assert(*s == -1);
    //     s[1] = -2;

    //     // inc / dec over stride boundary
    //     //      printf("ASSIGING AT THE EDGE//n");
    //     ss = s + n[N - 1] - 1;
    //     *ss = -3;
    //     // ss.print_current();
    //     assert(thrust::distance(s, ss) == n[N - 1] - 1);
    //     ++ss;
    //     // ss.print_current();
    //     *ss = -4;
    //     assert(thrust::distance(s, ss) == n[N - 1]);
    //     // printf("DECREMENT OVER EDGE//n");
    //     --ss;
    //     // ss.print_current();
    //     assert(*ss == -3);
    //     assert(thrust::distance(s, ss) == n[N - 1] - 1);
    //     // ss.print_current();

    //     // advance over stride boundary
    //     // printf("CHECKING ADVANCE//n");
    //     ss = s + n[N - 1];
    //     // ss.print_current();
    //     assert(*ss == -4);
    //     assert(thrust::distance(s, ss) == n[N - 1]);

    //     int subsize = n[0];
    //     for (int i = 1; i < N; i++) subsize *= n[i];
    //     // printf("subsize: %d//n", subsize);

    //     ss = s + (subsize - 1);
    //     // ss.print_current();
    //     assert(thrust::distance(s, ss) == subsize - 1);
    //     *ss = -5;
    //     ++ss; // last
    //     // ss.print_current();
    //     assert(thrust::distance(s, ss) == subsize);

    //     ss = s + subsize; // last
    //     // ss.print_current();
    //     assert(thrust::distance(s, ss) == subsize);

    //     if constexpr (N == 3) {
    //         ss = s + n[N - 2] * n[N - 1] - 1;
    //         assert(thrust::distance(s, ss) == n[N - 2] * n[N - 1] - 1);
    //         *ss = -6;
    //         ++ss;
    //         assert(thrust::distance(s, ss) == n[N - 2] * n[N - 1]);
    //         *ss = -7;
    //         --ss;
    //         assert(*ss == -6);
    //     }
    // }
    // thrust::copy(x.begin(), x.end(), v);
}

template struct transpose_test_cuda<double, 0, 1, 2>;
template struct transpose_test_cuda<double, 0, 2, 1>;
template struct transpose_test_cuda<double, 1, 0, 2>;
template struct transpose_test_cuda<double, 1, 2, 0>;
template struct transpose_test_cuda<double, 2, 0, 1>;
template struct transpose_test_cuda<double, 2, 1, 0>;
template struct transpose_test_cuda<double, 0, 1>;
template struct transpose_test_cuda<double, 1, 0>;
template struct transpose_test_cuda<float, 0, 1, 2>;
template struct transpose_test_cuda<float, 0, 2, 1>;
template struct transpose_test_cuda<float, 1, 0, 2>;
template struct transpose_test_cuda<float, 1, 2, 0>;
template struct transpose_test_cuda<float, 2, 0, 1>;
template struct transpose_test_cuda<float, 2, 1, 0>;
template struct transpose_test_cuda<float, 0, 1>;
template struct transpose_test_cuda<float, 1, 0>;
