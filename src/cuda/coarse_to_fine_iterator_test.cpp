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


#include "../coarse_to_fine_iterator_test.hpp"

#include "md_device_span.hpp"

template <typename T>
void test<T>::init(int fi0, int fi1, int ci0, int ci1, int ratio, const T* c_, T* f_)
{
    const auto ci = Ib{ci0, ci1}, fi = Ib{fi0, fi1};
    auto c = make_md_span(c_, ci);
    auto f = make_md_span(f_, fi);

    with_domain(fi)(f = c.fine(ratio));
}

template struct test<double>;
