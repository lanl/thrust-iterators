#include "submatrix_iterator.hpp"
#include "submatrix_iterator_test.hpp"

#include <cassert>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>

template <int N, typename T>
void submatrix_test_cuda<N, T>::init(T* v,
                                     const int (&sz)[N],
                                     const int (&lb)[N],
                                     const int (&ub)[N])
{
    int z = sz[0];
    for (int i = 1; i < N; i++) z *= sz[i];
    int n[N];
    for (int i = 0; i < N; i++) n[i] = ub[i] - lb[i] + 1;

    thrust::device_vector<T> x(v, v + z);
    printf("vector sz: %d\n", z);
    {
        auto s = make_submatrix(x.begin(), sz, lb, ub);
//        s.print_current();
        *s = -1;
//        s.print_current();
        assert(s[0] == -1);
        auto ss = s + 1;

        assert(thrust::distance(s, ss) == 1);

        ++s;
        --s;
        assert(*s == -1);
        s[1] = -2;

        // inc / dec over stride boundary
        //      printf("ASSIGING AT THE EDGE\n");
        ss = s + n[N - 1] - 1;
        *ss = -3;
        //ss.print_current();
        assert(thrust::distance(s, ss) == n[N - 1] - 1);
        ++ss;
        //ss.print_current();
        *ss = -4;
        assert(thrust::distance(s, ss) == n[N - 1]);
        //printf("DECREMENT OVER EDGE\n");
        --ss;
        //ss.print_current();
        assert(*ss == -3);
        assert(thrust::distance(s, ss) == n[N - 1] - 1);
        //ss.print_current();

        // advance over stride boundary
        //printf("CHECKING ADVANCE\n");
        ss = s + n[N - 1];
        //ss.print_current();
        assert(*ss == -4);
        assert(thrust::distance(s, ss) == n[N - 1]);

        int subsize = n[0];
        for (int i = 1; i < N; i++) subsize *= n[i];
        //printf("subsize: %d\n", subsize);

        ss = s + (subsize - 1);
        //ss.print_current();
        assert(thrust::distance(s, ss) == subsize - 1);
        *ss = -5;
        ++ss; // last
        //ss.print_current();
        assert(thrust::distance(s, ss) == subsize);

        ss = s + subsize; // last
        //ss.print_current();
        assert(thrust::distance(s, ss) == subsize);

        if constexpr (N == 3) {
            ss = s + n[N - 2] * n[N - 1] - 1;
            assert(thrust::distance(s, ss) == n[N - 2] * n[N - 1] - 1);
            *ss = -6;
            ++ss;
            assert(thrust::distance(s, ss) == n[N - 2] * n[N - 1]);
            *ss = -7;
            --ss;
            assert(*ss == -6);
        }
    }
    thrust::copy(x.begin(), x.end(), v);
}

template <int N, typename T>
void submatrix_test_cuda<N, T>::tabulate(T* v,
                                         const int (&sz)[N],
                                         const int (&lb)[N],
                                         const int (&ub)[N])
{
    int z = sz[0];
    for (int i = 1; i < N; i++) z *= sz[i];
    int n[N];
    for (int i = 0; i < N; i++) n[i] = ub[i] - lb[i] + 1;
    int n_sz = n[0];
    for (int i = 1; i < N; i++) n_sz *= n[i];

    thrust::device_vector<T> x(v, v + z);

    auto s = make_submatrix(x.begin(), sz, lb, ub);
    thrust::tabulate(s, s + n_sz, thrust::negate<T>());

    thrust::copy(x.begin(), x.end(), v);
}

template struct submatrix_test_cuda<2, double>;
template struct submatrix_test_cuda<3, double>;
template struct submatrix_test_cuda<2, float>;
template struct submatrix_test_cuda<3, float>;
