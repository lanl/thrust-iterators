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


#include "../coarse_to_fine_cuda.hpp"
#include "md_device_span.hpp"

template <typename T>
void coarse_to_fine<T>::copy(const int& ci0,
                             const int& ci1,
                             const int& cj0,
                             const int& cj1,
                             const int& ck0,
                             const int& ck1,
                             const int& fi0,
                             const int& fi1,
                             const int& fj0,
                             const int& fj1,
                             const int& fk0,
                             const int& fk1,
                             const int& axis,
                             const int& cbi0,
                             const int& cbi1,
                             const int& cbj0,
                             const int& cbj1,
                             const int& cbk0,
                             const int& cbk1,
                             const int& fbi0,
                             const int& fbi1,
                             const int& fbj0,
                             const int& fbj1,
                             const int& fbk0,
                             const int& fbk1,
                             const int& gcw,
                             const int* ratio,
                             const T* cdata_,
                             T* fdata_)
{
    auto c = make_md_span(cdata_, gcw, Kb{cbk0, cbk1}, Jb{cbj0, cbj1}, Ib{cbi0, cbi1});
    auto f = make_md_span(fdata_, gcw, Kb{fbk0, fbk1}, Jb{fbj0, fbj1}, Ib{fbi0, fbi1});

    // low and full bounds
    auto cil = Ib{ci0, ci0}, fil = Ib{fi0, fi0}, fi = Ib{fi0, fi1};
    auto cjl = Jb{cj0, cj0}, fjl = Jb{fj0, fj0}, fj = Jb{fj0, fj1};
    auto ckl = Kb{ck0, ck0}, fkl = Kb{fk0, fk0}, fk = Kb{fk0, fk1};

    switch (axis) {
    case 0:
        with_domain(fkl, fjl, fi)(f = c.fine(ratio[axis], ckl, cjl));
        break;
    case 1:
        with_domain(fkl, fj, fil)(f = c.fine(ratio[axis], ckl, cil));
        break;
    case 2:
        with_domain(fk, fjl, fil)(f = c.fine(ratio[axis], cjl, cil));
        break;
    }
}

template <typename T>
void coarse_to_fine<T>::copy_corner(const int& ci0,
                                    const int& ci1,
                                    const int& cj0,
                                    const int& cj1,
                                    const int& ck0,
                                    const int& ck1,
                                    const int& fi0,
                                    const int& fi1,
                                    const int& fj0,
                                    const int& fj1,
                                    const int& fk0,
                                    const int& fk1,
                                    const int& cbi0,
                                    const int& cbi1,
                                    const int& cbj0,
                                    const int& cbj1,
                                    const int& cbk0,
                                    const int& cbk1,
                                    const int& fbi0,
                                    const int& fbi1,
                                    const int& fbj0,
                                    const int& fbj1,
                                    const int& fbk0,
                                    const int& fbk1,
                                    const int& gcw,
                                    const T* cdata_,
                                    T* fdata_)
{
    auto c = make_md_span(cdata_, gcw, Kb{cbk0, cbk1}, Jb{cbj0, cbj1}, Ib{cbi0, cbi1});
    auto f = make_md_span(fdata_, gcw, Kb{fbk0, fbk1}, Jb{fbj0, fbj1}, Ib{fbi0, fbi1});

    f.at(fk0, fj0, fi0) = c.at(ck0, cj0, ci0);
}

template struct coarse_to_fine<float>;
template struct coarse_to_fine<double>;
