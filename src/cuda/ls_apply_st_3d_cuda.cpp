#include "../ls_apply_st_3d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void ls_apply_st_3d_cuda<T>::at_point(const int& i0,
                                      const int& i1,
                                      const int& j0,
                                      const int& j1,
                                      const int& k0,
                                      const int& k1,
                                      const int& i,
                                      const int& j,
                                      const int& k,
                                      const int& stencil_sz,
                                      const int* offsets_,
                                      const T* st_,
                                      const T& a,
                                      const T& b,
                                      const int& fgcw,
                                      const T* f_,
                                      const int& ugcw,
                                      const T* u_,
                                      const int& rgcw,
                                      const T* r_)
{
    const auto ib = Ib{i0, i1};
    const auto jb = Jb{j0, j1};
    const auto kb = Kb{k0, k1};
    const auto wb = Wb{1, stencil_sz};

    auto st = make_md_span(st_, kb, jb, ib, wb);
    auto f = make_md_span(f_, fgcw, kb, jb, ib);
    auto u = make_md_span(u_, ugcw, kb, jb, ib);
    auto r = make_md_span(r_, rgcw, kb, jb, ib);
    auto o = make_md_span(offsets_, 3, wb); // this isn't right

    with_domain(Kb{k, k}, Jb{j, j}, Ib{i, i})(r = b * f + a * dot(st, u.offset(o), wb));
}
