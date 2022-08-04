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


#include "../cd_stencil_3d_cuda.hpp"

#include "md_device_span.hpp"

template <typename T>
void cd_stencil_3d_cuda<T>::offdiag(const int& i0,
                                    const int& j0,
                                    const int& k0,
                                    const int& i1,
                                    const int& j1,
                                    const int& k1,
                                    const int& bi0,
                                    const int& bj0,
                                    const int& bk0,
                                    const int& bi1,
                                    const int& bj1,
                                    const int& bk1,
                                    const T* dx,
                                    const T& beta,
                                    const T* b0_,
                                    const T* b1_,
                                    const T* b2_,
                                    const int& sgcw,
                                    T* st_)
{
    const auto i = Ib{i0, i1}, bi = Ib{bi0, bi1};
    const auto j = Jb{j0, j1}, bj = Jb{bj0, bj1};
    const auto k = Kb{k0, k1}, bk = Kb{bk0, bk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto b0 = make_md_span(b0_, bk, bj, bi + 1);
    auto b1 = make_md_span(b1_, bi, bk, bj + 1);
    auto b2 = make_md_span(b2_, bj, bi, bk + 1);

    T d0 = -beta / (dx[0] * dx[0]);
    T d1 = -beta / (dx[1] * dx[1]);
    T d2 = -beta / (dx[2] * dx[2]);

    with_domain(st.window(), k, j, i)(st(1, 2) = d0 * b0.stencil_x(),
                                      st(3, 4) = d1 * b1.stencil_y(),
                                      st(5, 6) = d2 * b2.stencil_z());

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_3d_cuda<T>::poisson_offdiag(const int& i0,
                                            const int& j0,
                                            const int& k0,
                                            const int& i1,
                                            const int& j1,
                                            const int& k1,
                                            const T* dx,
                                            const T& beta,
                                            const int& sgcw,
                                            T* st_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    T d0 = -beta / (dx[0] * dx[0]);
    T d1 = -beta / (dx[1] * dx[1]);
    T d2 = -beta / (dx[2] * dx[2]);

    with_domain(st.window(), k, j, i)(st(1, 2) = d0, st(3, 4) = d1, st(5, 6) = d2);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_3d_cuda<T>::v1diag(const int& i0,
                                   const int& j0,
                                   const int& k0,
                                   const int& i1,
                                   const int& j1,
                                   const int& k1,
                                   const int& ai0,
                                   const int& aj0,
                                   const int& ak0,
                                   const int& ai1,
                                   const int& aj1,
                                   const int& ak1,
                                   const T& alpha,
                                   const T* a_,
                                   const int& sgcw,
                                   T* st_)
{
    const auto i = Ib{i0, i1}, ai = Ib{ai0, ai1};
    const auto j = Jb{j0, j1}, aj = Jb{aj0, aj1};
    const auto k = Kb{k0, k1}, ak = Kb{ak0, ak1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto a = make_md_span(a_, ak, aj, ai);

    with_domain(st.window(), k, j, i)(
        st(0) = -(st(1) + st(2) + st(3) + st(4) + st(5) + st(6)) + alpha * a);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_3d_cuda<T>::v2diag(const int& i0,
                                   const int& j0,
                                   const int& k0,
                                   const int& i1,
                                   const int& j1,
                                   const int& k1,
                                   const T& alpha,
                                   const int& sgcw,
                                   T* st_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    with_domain(st.window(), k, j, i)(
        st(0) = -(st(1) + st(2) + st(3) + st(4) + st(5) + st(6)) + alpha);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_3d_cuda<T>::poisson_diag(const int& i0,
                                         const int& j0,
                                         const int& k0,
                                         const int& i1,
                                         const int& j1,
                                         const int& k1,
                                         const int& sgcw,
                                         T* st_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    with_domain(st.window(), k, j, i)(
        st(0) = -(st(1) + st(2) + st(3) + st(4) + st(5) + st(6)));

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_diag(const int& i0,
                                     const int& j0,
                                     const int& k0,
                                     const int& i1,
                                     const int& j1,
                                     const int& k1,
                                     const int& pi0,
                                     const int& pj0,
                                     const int& pk0,
                                     const int& pi1,
                                     const int& pj1,
                                     const int& pk1,
                                     const int& dir,
                                     const int& side,
                                     const int& btype,
                                     const int& exOrder,
                                     const T* dx,
                                     const T& beta,
                                     const T* b0_,
                                     const T* b1_,
                                     const T* b2_,
                                     const int& sgcw,
                                     T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto b0 = make_md_span(b0_, k, j, i + 1);
    auto b1 = make_md_span(b1_, i, k, j + 1);
    auto b2 = make_md_span(b2_, j, i, k + 1);

    T h{dx[dir]};

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto bt4 = [&](T f, auto&& b, auto&&... p) mutable {
        if (exOrder == 2) {
            with_domain(st.window(),
                        FWD(p)...)(st(0) = st(0) - (f * b) * (3 * h + 16 * b));
            st.copy_to(st_);
        }
    };

    auto bt0 = [&](T f, auto&& b, auto&&... p) mutable {
        if (exOrder == 2) {
            with_domain(st.window(), FWD(p)...)(st(0) = st(0) + (f * b));
            st.copy_to(st_);
        }
    };

    switch (btype) {
    case 4:
        switch (T f = beta / h; dir) {
        case 0:
            bt4(f, b0.shift_x(side), pk, pj, pil);
            break;
        case 1:
            bt4(f, b1.shift_y(side), pk, pjl, pi);
            break;
        case 2:
            bt4(f, b2.shift_z(side), pkl, pj, pi);
            break;
        }
        break;
    case 0:
        switch (T f = beta / (3 * h * h); dir) {
        case 0:
            bt0(f, b0, pk, pj, pil);
            break;
        case 1:
            bt0(f, b1, pk, pjl, pi);
            break;
        case 2:
            bt0(f, b2, pkl, pj, pi);
            break;
        } // switch (dir)
        break;
    } // switch (btype)
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_poisson_diag(const int& i0,
                                             const int& j0,
                                             const int& k0,
                                             const int& i1,
                                             const int& j1,
                                             const int& k1,
                                             const int& pi0,
                                             const int& pj0,
                                             const int& pk0,
                                             const int& pi1,
                                             const int& pj1,
                                             const int& pk1,
                                             const int& dir,
                                             const int& side,
                                             const int& btype,
                                             const int& exOrder,
                                             const T* dx,
                                             const T& beta,
                                             const int& sgcw,
                                             T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    T h{dx[dir]};

    if (exOrder != 2) return;

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto bt4 = [&](T f, auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(0) = st(0) - (f * (3 * h + 16)));
        st.copy_to(st_);
    };
    auto bt0 = [&](T f, auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(0) = st(0) + f);
        st.copy_to(st_);
    };

    switch (btype) {
    case 4:
        switch (T f = beta / h; dir) {
        case 0:
            bt4(f, pk, pj, pil);
            break;
        case 1:
            bt4(f, pk, pjl, pi);
            break;
        case 2:
            bt4(f, pkl, pj, pi);
            break;
        }
        break;
    case 0:
        switch (T f = beta / (3 * h * h); dir) {
        case 0:
            bt0(f, pk, pj, pil);
            break;
        case 1:
            bt0(f, pk, pjl, pi);
            break;
        case 2:
            bt0(f, pkl, pj, pi);
            break;
        }
        break;
    } // switch (btype)
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_cf_diag(const int& i0,
                                        const int& j0,
                                        const int& k0,
                                        const int& i1,
                                        const int& j1,
                                        const int& k1,
                                        const int& pi0,
                                        const int& pj0,
                                        const int& pk0,
                                        const int& pi1,
                                        const int& pj1,
                                        const int& pk1,
                                        const int& r,
                                        const int& dir,
                                        const int& side,
                                        const int& intOrder,
                                        const T* dx,
                                        const T& beta,
                                        const T* b0_,
                                        const T* b1_,
                                        const T* b2_,
                                        const int& sgcw,
                                        T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};
    const T dr = 2.0 * (r - 1.0) / (r + 1.0);

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto b0 = make_md_span(b0_, k, j, i + 1);
    auto b1 = make_md_span(b1_, i, k, j + 1);
    auto b2 = make_md_span(b2_, j, i, k + 1);

    T h{dx[dir]};
    T f = beta / (h * h);

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto func = [&](auto&& b, auto&&... p) mutable {
        if (intOrder == 1)
            with_domain(st.window(), FWD(p)...)(st(0) = st(0) - (f / 3) * b);
        else if (intOrder == 2)
            with_domain(st.window(), FWD(p)...)(st(0) = st(0) - (f * dr) * b);

        st.copy_to(st_);
    };

    switch (T f = beta / (h * h); dir) {
    case 0:
        func(b0.shift_x(side), pk, pj, pil);
        break;
    case 1:
        func(b1.shift_y(side), pk, pjl, pi);
        break;
    case 2:
        func(b2.shift_z(side), pkl, pj, pi);
        break;
    }
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_poisson_cf_diag(const int& i0,
                                                const int& j0,
                                                const int& k0,
                                                const int& i1,
                                                const int& j1,
                                                const int& k1,
                                                const int& pi0,
                                                const int& pj0,
                                                const int& pk0,
                                                const int& pi1,
                                                const int& pj1,
                                                const int& pk1,
                                                const int& r,
                                                const int& dir,
                                                const int& side,
                                                const int& intOrder,
                                                const T* dx,
                                                const T& beta,
                                                const int& sgcw,
                                                T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};
    const T dr = 2.0 * (r - 1.0) / (r + 1.0);

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    T h{dx[dir]};
    T f = beta / (h * h);

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto func = [&](auto&&... p) mutable {
        if (intOrder == 1)
            with_domain(st.window(), FWD(p)...)(st(0) = st(0) - (f / 3));
        else if (intOrder == 2)
            with_domain(st.window(), FWD(p)...)(st(0) = st(0) - (f * dr));
        st.copy_to(st_);
    };

    switch (dir) {
    case 0:
        func(pk, pj, pil);
        break;
    case 1:
        func(pk, pjl, pi);
        break;
    case 2:
        func(pkl, pj, pi);
        break;
    }
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_offdiag(const int& i0,
                                        const int& j0,
                                        const int& k0,
                                        const int& i1,
                                        const int& j1,
                                        const int& k1,
                                        const int& pi0,
                                        const int& pj0,
                                        const int& pk0,
                                        const int& pi1,
                                        const int& pj1,
                                        const int& pk1,
                                        const int& dir,
                                        const int& side,
                                        const int& btype,
                                        const int& exOrder,
                                        const T* dx,
                                        const T& dir_factor,
                                        const T& neu_factor,
                                        const T& beta,
                                        const T* b0_,
                                        const T* b1_,
                                        const T* b2_,
                                        const int& sgcw,
                                        T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto b0 = make_md_span(b0_, k, j, i + 1);
    auto b1 = make_md_span(b1_, i, k, j + 1);
    auto b2 = make_md_span(b2_, j, i, k + 1);

    T h{dx[dir]};

    int u = 2 * dir + 1 + side;
    int v = 2 * dir + 2 - side;

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto bt4 = [&](auto&& b, auto&&... p) mutable {
        if (exOrder == 1) {
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * (2 * h / (4 * b + h)));
        } else if (exOrder == 2) {
            auto f = beta * b / (h * (3 * h + 16 * b));
            with_domain(st.window(), FWD(p)...)(st(u) = -9 * f, st(v) = st(v) - f);
        }
        st.copy_to(st_);
    };

    auto bt0 = [&](auto&&... p) mutable {
        if (exOrder == 1) {
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * dir_factor);
        } else if (exOrder == 2) {
            with_domain(st.window(), FWD(p)...)(st(v) = st(v) + st(u) / 3,
                                                st(u) = st(u) * (8.0 / 3.0));
        }
        st.copy_to(st_);
    };

    auto bt1 = [&](auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(u) = 0.0);
        st.copy_to(st_);
    };

    switch (btype) {
    case 4:
        switch (dir) {
        case 0:
            bt4(b0.shift_x(side), pk, pj, pil);
            return;
        case 1:
            bt4(b1.shift_y(side), pk, pjl, pi);
            return;
        case 2:
            bt4(b2.shift_z(side), pkl, pj, pi);
            return;
        }
        break;
    case 0:
        switch (dir) {
        case 0:
            bt0(pk, pj, pil);
            return;
        case 1:
            bt0(pk, pjl, pi);
            return;
        case 2:
            bt0(pkl, pj, pi);
            return;
        }
        break;
    case 1:
        switch (dir) {
        case 0:
            bt1(pk, pj, pil);
            return;
        case 1:
            bt1(pk, pjl, pi);
            return;
        case 2:
            bt1(pkl, pj, pi);
            return;
        }
        break;
    } // switch (btype)
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_poisson_offdiag(const int& i0,
                                                const int& j0,
                                                const int& k0,
                                                const int& i1,
                                                const int& j1,
                                                const int& k1,
                                                const int& pi0,
                                                const int& pj0,
                                                const int& pk0,
                                                const int& pi1,
                                                const int& pj1,
                                                const int& pk1,
                                                const int& dir,
                                                const int& side,
                                                const int& btype,
                                                const int& exOrder,
                                                const T* dx,
                                                const T& dir_factor,
                                                const T& neu_factor,
                                                const T& beta,
                                                const int& sgcw,
                                                T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    T h{dx[dir]};

    int u = 2 * dir + 1 + side;
    int v = 2 * dir + 2 - side;

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto bt4 = [&](auto&&... p) mutable {
        if (exOrder == 1) {
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * (2 * h / (4 + h)));
        } else if (exOrder == 2) {
            T f = beta / (h * (3 * h + 16));
            with_domain(st.window(), FWD(p)...)(st(u) = -9 * f, st(v) = st(v) - f);
        }
        st.copy_to(st_);
    };

    auto bt0 = [&](auto&&... p) mutable {
        if (exOrder == 1) {
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * dir_factor);
        } else if (exOrder == 2) {
            with_domain(st.window(), FWD(p)...)(st(v) = st(v) + st(u) / 3,
                                                st(u) = st(u) * (8.0 / 3.0));
        }
        st.copy_to(st_);
    };

    auto bt1 = [&](auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(u) = 0.0);
        st.copy_to(st_);
    };

    switch (btype) {
    case 4:
        switch (dir) {
        case 0:
            bt4(pk, pj, pil);
            return;
        case 1:
            bt4(pk, pjl, pi);
            return;
        case 2:
            bt4(pkl, pj, pi);
            return;
        }
        break;
    case 0:
        switch (dir) {
        case 0:
            bt0(pk, pj, pil);
            return;
        case 1:
            bt0(pk, pjl, pi);
            return;
        case 2:
            bt0(pkl, pj, pi);
            return;
        }
        break;
    case 1:
        switch (dir) {
        case 0:
            bt1(pk, pj, pil);
            return;
        case 1:
            bt1(pk, pjl, pi);
            return;
        case 2:
            bt1(pkl, pj, pi);
            return;
        }
        break;
    } // switch (btype)
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_cf_offdiag(const int& i0,
                                           const int& j0,
                                           const int& k0,
                                           const int& i1,
                                           const int& j1,
                                           const int& k1,
                                           const int& ci0,
                                           const int& cj0,
                                           const int& ck0,
                                           const int& ci1,
                                           const int& cj1,
                                           const int& ck1,
                                           const int& r,
                                           const int& dir,
                                           const int& side,
                                           const int& intOrder,
                                           const int& sgcw,
                                           T* st_)
{
    const auto i = Ib{i0, i1}, ci = Ib{ci0, ci1};
    const auto j = Jb{j0, j1}, cj = Jb{cj0, cj1};
    const auto k = Kb{k0, k1}, ck = Kb{ck0, ck1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    // localize indicies
    const auto cil = Ib{ci0 - 2 * side + 1, ci0 - 2 * side + 1};
    const auto cjl = Jb{cj0 - 2 * side + 1, cj0 - 2 * side + 1};
    const auto ckl = Kb{ck0 - 2 * side + 1, ck0 - 2 * side + 1};

    int u = 2 * dir + 1 + side;
    int v = 2 * dir + 2 - side;

    auto func = [&](auto&&... p) mutable {
        if (intOrder == 1)
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * (2.0 / 3.0));
        else if (intOrder == 2)
            with_domain(st.window(),
                        FWD(p)...)(st(v) = st(v) - st(u) * ((r - 1.0) / (r + 3.0)),
                                   st(u) = st(u) * (8.0 / ((r + 1.0) * (r + 3.0))));

        st.copy_to(st_);
    };

    switch (dir) {
    case 0:
        func(ck, cj, cil);
        break;
    case 1:
        func(ck, cjl, ci);
        break;
    case 2:
        func(ckl, cj, ci);
        break;
    }
}

template <typename T>
void cd_stencil_3d_cuda<T>::readj_offdiag(const int& i0,
                                          const int& j0,
                                          const int& k0,
                                          const int& i1,
                                          const int& j1,
                                          const int& k1,
                                          const int& pi0,
                                          const int& pj0,
                                          const int& pk0,
                                          const int& pi1,
                                          const int& pj1,
                                          const int& pk1,
                                          const int& dir,
                                          const int& side,
                                          const int& sgcw,
                                          T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);

    int u = 2 * dir + 1 + side;

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto f = [&](auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(u) = 0.0);
        st.copy_to(st_);
    };

    switch (dir) {
    case 0:
        f(pk, pj, pil);
        return;
    case 1:
        f(pk, pjl, pi);
        return;
    case 2:
        f(pkl, pj, pi);
        return;
    }
}

template <typename T>
void cd_stencil_3d_cuda<T>::adj_cf_bdryrhs(const int& i0,
                                           const int& j0,
                                           const int& k0,
                                           const int& i1,
                                           const int& j1,
                                           const int& k1,
                                           const int& pi0,
                                           const int& pj0,
                                           const int& pk0,
                                           const int& pi1,
                                           const int& pj1,
                                           const int& pk1,
                                           const int& dir,
                                           const int& side,
                                           const int& sgcw,
                                           const T* st_,
                                           const int& gcw,
                                           const T* u_,
                                           T* rhs_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto j = Jb{j0, j1}, pj = Jb{pj0, pj1};
    const auto k = Kb{k0, k1}, pk = Kb{pk0, pk1};
    const auto w = Wb{0, 6};

    auto st = make_md_span(st_, sgcw, k, j, i, w);
    auto u = make_md_span(u_, gcw, k, j, i);
    auto rhs = make_md_span(rhs_, k, j, i);

    int ui = 2 * dir + 1 + side;
    int s = 2 * side - 1;

    // localize indicies
    const auto pil = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto pjl = Jb{pj0 - 2 * side + 1, pj0 - 2 * side + 1};
    const auto pkl = Kb{pk0 - 2 * side + 1, pk0 - 2 * side + 1};

    auto f = [&](auto&& v, auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(rhs -= st(ui) * v);
        rhs.copy_to(rhs_);
    };

    switch (dir) {
    case 0:
        f(u.shift_x(s), pk, pj, pil);
        return;
    case 1:
        f(u.shift_y(s), pk, pjl, pi);
        return;
    case 2:
        f(u.shift_z(s), pkl, pj, pi);
        return;
    }
}

template struct cd_stencil_3d_cuda<float>;
template struct cd_stencil_3d_cuda<double>;
