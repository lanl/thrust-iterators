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
