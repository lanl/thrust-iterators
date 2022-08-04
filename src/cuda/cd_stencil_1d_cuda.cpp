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


#include "../cd_stencil_1d_cuda.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

#include "md_device_span.hpp"
#include <cassert>

// For now these are wrappers around the device kernels that simply handle data transfer.
// The current assumption is that all data coming in is host data.
template <typename T>
void cd_stencil_1d_cuda<T>::offdiag(const int& i0,
                                    const int& i1,
                                    const int& bi0,
                                    const int& bi1,
                                    const T* dx,
                                    const T& beta,
                                    const T* b0_,
                                    const int& sgcw,
                                    T* st_)
{

    const auto i = Ib{i0, i1};
    const auto b = Ib{bi0, bi1};
    const auto w = Wb{0, 2};

    auto b0 = make_md_span(b0_, b + 1);
    auto st = make_md_span(st_, sgcw, i, w);

    const T d0 = -beta / (*dx * *dx);

    with_domain(st.window(), i)(st(1, 2) = d0 * b0.stencil_x());

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_1d_cuda<T>::poisson_offdiag(
    const int& i0, const int& i1, const T* dx, const T& beta, const int& sgcw, T* st_)
{

    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    T d0 = -beta / (*dx * *dx);

    with_domain(st.window(), i)(st(1) = d0, st(2) = d0);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_1d_cuda<T>::v1diag(const int& i0,
                                   const int& i1,
                                   const int& ai0,
                                   const int& ai1,
                                   const T& alpha,
                                   const T* a_,
                                   const int& sgcw,
                                   T* st_)
{
    const auto i = Ib{i0, i1};
    const auto ai = Ib{ai0, ai1};
    const auto w = Wb{0, 2};

    auto a = make_md_span(a_, ai);
    auto st = make_md_span(st_, sgcw, i, w);

    with_domain(st.window(), i)(st(0) = -(st(1) + st(2)) + alpha * a);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_1d_cuda<T>::v2diag(
    const int& i0, const int& i1, const T& alpha, const int& sgcw, T* st_)
{
    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    with_domain(st.window(), i)(st(0) = -(st(1) + st(2)) + alpha);

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_1d_cuda<T>::poisson_diag(const int& i0,
                                         const int& i1,
                                         const int& sgcw,
                                         T* st_)
{

    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    with_domain(st.window(), i)(st(0) = -(st(1) + st(2)));

    st.copy_to(st_);
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_diag(const int& i0,
                                     const int& i1,
                                     const int& pi0,
                                     const int& pi1,
                                     const int& side,
                                     const int& btype,
                                     const int& exOrder,
                                     const T* dx,
                                     const T& beta,
                                     const T* b0_,
                                     const int& sgcw,
                                     T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);
    auto b0 = make_md_span(b0_, i + 1);

    T h{dx[0]};

    if (btype == 4) {
        T f = beta / h;
        const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
        if (exOrder == 2) {
            auto b = b0.shift_x(side);
            with_domain(st.window(), pi)(st(0) = st(0) - (f * b) * (3 * h + 16 * b));
            st.copy_to(st_);
        }
    }
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_poisson_diag(const int& i0,
                                             const int& i1,
                                             const int& pi0,
                                             const int& pi1,
                                             const int& side,
                                             const int& btype,
                                             const int& exOrder,
                                             const T* dx,
                                             const T& beta,
                                             const int& sgcw,
                                             T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0, pi1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    T h{dx[0]};

    if (btype == 4 && exOrder == 2) {
        const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};

        T f = beta / h;
        with_domain(st.window(), pi)(st(0) = st(0) - (f * (3 * h + 16)));
        st.copy_to(st_);
    }
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_cf_diag(const int& i0,
                                        const int& i1,
                                        const int& pi0,
                                        const int& pi1,
                                        const int& r,
                                        const int& side,
                                        const int& intOrder,
                                        const T* dx,
                                        const T& beta,
                                        const T* b0_,
                                        const int& sgcw,
                                        T* st_)
{
    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};
    const T dr = 2.0 * (r - 1.0) / (r + 1.0);

    auto st = make_md_span(st_, sgcw, i, w);
    auto b0 = make_md_span(b0_, i + 1);

    T h{dx[0]};
    T f = beta / (h * h);

    const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    auto b = b0.shift_x(side);

    if (intOrder == 1)
        with_domain(st.window(), pi)(st(0) = st(0) - (f / 3) * b);
    else if (intOrder == 2)
        with_domain(st.window(), pi)(st(0) = st(0) - (f * dr) * b);

    st.copy_to(st_);
    return;
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_poisson_cf_diag(const int& i0,
                                                const int& i1,
                                                const int& pi0,
                                                const int& pi1,
                                                const int& r,
                                                const int& side,
                                                const int& intOrder,
                                                const T* dx,
                                                const T& beta,
                                                const int& sgcw,
                                                T* st_)
{
    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};
    const T dr = 2.0 * (r - 1.0) / (r + 1.0);

    auto st = make_md_span(st_, sgcw, i, w);

    T h{dx[0]};
    T f = beta / (h * h);

    const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};

    if (intOrder == 1)
        with_domain(st.window(), pi)(st(0) = st(0) - (f / 3));
    else if (intOrder == 2)
        with_domain(st.window(), pi)(st(0) = st(0) - (f * dr));

    st.copy_to(st_);
    return;
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_offdiag(const int& i0,
                                        const int& i1,
                                        const int& pi0,
                                        const int& pi1,
                                        const int& side,
                                        const int& btype,
                                        const int& exOrder,
                                        const T* dx,
                                        const T& dir_factor,
                                        const T& neu_factor,
                                        const T& beta,
                                        const T* b0_,
                                        const int& sgcw,
                                        T* st_)
{
    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);
    auto b0 = make_md_span(b0_, i + 1);

    T h{dx[0]};

    int u = 1 + side;
    int v = 2 - side;

    const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};

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
        return bt4(b0.shift_x(side), pi);
    case 0:
        return bt0(pi);
    case 1:
        return bt1(pi);
    }
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_poisson_offdiag(const int& i0,
                                                const int& i1,
                                                const int& pi0,
                                                const int& pi1,
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
    const auto i = Ib{i0, i1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    T h{dx[0]};

    int u = 1 + side;
    int v = 2 - side;

    const auto pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};

    auto bt4 = [&](auto&&... p) mutable {
        if (exOrder == 1) {
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * (2 * h / (4 + h)));
        } else if (exOrder == 2) {
            auto f = beta / (h * (3 * h + 16));
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
        return bt4(pi);
    case 0:
        return bt0(pi);
    case 1:
        return bt1(pi);
    }
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_cf_offdiag(const int& i0,
                                           const int& i1,
                                           const int& ci0,
                                           const int& ci1,
                                           const int& r,
                                           const int& side,
                                           const int& intOrder,
                                           const int& sgcw,
                                           T* st_)
{
    const auto i = Ib{i0, i1}, ci = Ib{ci0 - 2 * side + 1, ci0 - 2 * side + 1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    int u = 1 + side;
    int v = 2 - side;

    auto func = [&](auto&&... p) mutable {
        if (intOrder == 1)
            with_domain(st.window(), FWD(p)...)(st(u) = st(u) * (2.0 / 3.0));
        else if (intOrder == 2)
            with_domain(st.window(),
                        FWD(p)...)(st(v) = st(v) - st(u) * ((r - 1.0) / (r + 3.0)),
                                   st(u) = st(u) * (8.0 / ((r + 1.0) * (r + 3.0))));

        st.copy_to(st_);
    };

    func(ci);
}

template <typename T>
void cd_stencil_1d_cuda<T>::readj_offdiag(const int& i0,
                                          const int& i1,
                                          const int& pi0,
                                          const int& pi1,
                                          const int& side,
                                          const int& sgcw,
                                          T* st_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);

    int u = 1 + side;

    auto f = [&](auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(st(u) = 0.0);
        st.copy_to(st_);
    };

    f(pi);
}

template <typename T>
void cd_stencil_1d_cuda<T>::adj_cf_bdryrhs(const int& i0,
                                           const int& i1,
                                           const int& pi0,
                                           const int& pi1,
                                           const int& side,
                                           const int& sgcw,
                                           const T* st_,
                                           const int& gcw,
                                           const T* u_,
                                           T* rhs_)
{
    const auto i = Ib{i0, i1}, pi = Ib{pi0 - 2 * side + 1, pi0 - 2 * side + 1};
    const auto w = Wb{0, 2};

    auto st = make_md_span(st_, sgcw, i, w);
    auto u = make_md_span(u_, gcw, i);
    auto rhs = make_md_span(rhs_, i);

    int ui = 1 + side;
    int s = 2 * side - 1;

    auto f = [&](auto&& v, auto&&... p) mutable {
        with_domain(st.window(), FWD(p)...)(rhs -= st(ui) * v);
        rhs.copy_to(rhs_);
    };

    f(u.shift_x(s), pi);
}

template struct cd_stencil_1d_cuda<double>;
template struct cd_stencil_1d_cuda<float>;
