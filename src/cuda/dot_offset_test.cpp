#include "../dot_offset_test.hpp"
#include "md_device_span.hpp"

template <typename T>
void dot_offset_test<T>::init(const T* st_,
                              const int& sgcw,
                              const T* u_,
                              const int& ugcw,
                              T* v_,
                              const int& vgcw,
                              int* offset,
                              const int& i0,
                              const int& i1,
                              const int& w1)
{

    const auto i = Ib{i0, i1};
    const auto w = Wb{0, w1 - 1};
    auto st = make_md_span(st_, sgcw, i, w);
    auto u = make_md_span(u_, ugcw, i);
    auto v = make_md_span(v_, vgcw, i);
    auto o = thrust::device_ptr<int>(offset);

    with_domain(i)(v = st.dot(u.offset(o)));
}

template <typename T>
void dot_offset_test<T>::init(const T* st_,
                              const int& sgcw,
                              const T* u_,
                              const int& ugcw,
                              T* v_,
                              const int& vgcw,
                              int* offset,
                              const int& i0,
                              const int& i1,
                              const int& j0,
                              const int& j1,
                              const int& w1)
{

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto w = Wb{0, w1 - 1};
    auto st = make_md_span(st_, sgcw, j, i, w);
    auto u = make_md_span(u_, ugcw, j, i);
    auto v = make_md_span(v_, vgcw, j, i);
    auto o = thrust::device_ptr<int>(offset);

    with_domain(j, i)(v = st.dot(u.offset(o)));
}

template <typename T>
void dot_offset_test<T>::init(const T* st_,
                              const int& sgcw,
                              const T* u_,
                              const int& ugcw,
                              T* v_,
                              const int& vgcw,
                              int* offset,
                              const int& i0,
                              const int& i1,
                              const int& j0,
                              const int& j1,
                              const int& k0,
                              const int& k1,
                              const int& w1)
{

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};
    const auto w = Wb{0, w1 - 1};
    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto u = make_md_span(u_, ugcw, k, j, i);
    auto v = make_md_span(v_, vgcw, k, j, i);
    auto o = thrust::device_ptr<int>(offset);

    with_domain(k, j, i)(v = st.dot(u.offset(o)));
}

template struct dot_offset_test<double>;
