#include "../coarse_to_fine_cuda.hpp"
#include "md_lazy_vector.hpp"

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
    auto c = make_vec(cdata_, gcw, Kb{cbk0, cbk1}, Jb{cbj0, cbj1}, Ib{cbi0, cbi1});
    auto f = make_vec(fdata_, gcw, Kb{fbk0, fbk1}, Jb{fbj0, fbj1}, Ib{fbi0, fbi1});

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

    f.copy_to(fdata_);
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
    auto c = make_vec(cdata_, gcw, Kb{cbk0, cbk1}, Jb{cbj0, cbj1}, Ib{cbi0, cbi1});
    auto f = make_vec(fdata_, gcw, Kb{fbk0, fbk1}, Jb{fbj0, fbj1}, Ib{fbi0, fbi1});

    f.at(fk0, fj0, fi0) = c.at(ck0, cj0, ci0);

    f.copy_to(fdata_);
}

template struct coarse_to_fine<float>;
template struct coarse_to_fine<double>;
