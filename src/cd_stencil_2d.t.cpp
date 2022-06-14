#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes2d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_stencil_2d_cuda.hpp"
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

template <typename T>
void compare(const std::vector<T>& t, const std::vector<T>& u)
{
    REQUIRE_THAT(t, Approx(u));
}

template <typename T>
void compare(const std::vector<T>& t, const thrust::device_vector<T>& u)
{
    compare<T>(t, to_std(u));
}

constexpr auto w = Wb{0, 4};

template <typename T>
using coeffs = cd_stencil_2d_cuda<T>;

TEST_CASE("offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 40;
    const int i1 = 31, j1 = 78;
    const int bi0 = i0 - pick(1, 3), bj0 = j0 - pick(1, 3);
    const int bi1 = i1 + pick(2, 4), bj1 = j1 + pick(2, 4);

    const std::array dx{pick(0.1), pick(0.1)};
    const T beta = pick(0.2, 10.0);
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1}, bi = Ib{bi0, bi1};
    const auto j = Jb{j0, j1}, bj = Jb{bj0, bj1};

    auto b0 = make_md_vec<T>(bj, bi + 1);
    auto b1 = make_md_vec<T>(bi, bj + 1);
    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, j, i, w);

    b0.fill_random();
    b1.fill_random();

    celldiffusionoffdiag2d_(i0,
                            j0,
                            i1,
                            j1,
                            bi0,
                            bj0,
                            bi1,
                            bj1,
                            dx.data(),
                            beta,
                            b0.host_data(),
                            b1.host_data(),
                            sgcw,
                            st.host_data());

    coeffs<T>::offdiag(i0,
                       j0,
                       i1,
                       j1,
                       bi0,
                       bj0,
                       bi1,
                       bj1,
                       dx.data(),
                       beta,
                       b0.data(),
                       b1.data(),
                       sgcw,
                       st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 15;
    const int i1 = 21, j1 = 35;
    const std::array dx{0.5, 0.1};
    const T beta = pick(1.0, 10.0);
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, j, i, w);

    cellpoissonoffdiag2d_(i0, j0, i1, j1, dx.data(), beta, sgcw, st.host_data());

    coeffs<T>::poisson_offdiag(i0, j0, i1, j1, dx.data(), beta, sgcw, st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("v1diag")
{

    using T = double;
    randomize();

    const int i0 = 10, j0 = 15;
    const int i1 = 21, j1 = 35;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);
    const int ai0 = i0 - pick(1, 3), aj0 = j0 - pick(1, 3);
    const int ai1 = i1 + pick(2, 4), aj1 = j1 + pick(2, 4);

    const auto i = Ib{i0, i1}, ai = Ib{ai0, ai1};
    const auto j = Jb{j0, j1}, aj = Jb{aj0, aj1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto a = make_md_vec<T>(aj, ai);

    a.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv1diag2d_(
        i0, j0, i1, j1, ai0, aj0, ai1, aj1, alpha, a.host_data(), sgcw, st.host_data());

    coeffs<T>::v1diag(i0,
                      j0,
                      i1,
                      j1,
                      ai0,
                      aj0,
                      ai1,
                      aj1,
                      alpha,
                      a.data(),
                      sgcw,
                      thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("v2diag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 15;
    const int i1 = 21, j1 = 35;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv2diag2d_(i0, j0, i1, j1, alpha, sgcw, st.host_data());

    coeffs<T>::v2diag(
        i0, j0, i1, j1, alpha, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson diag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 15;
    const int i1 = 21, j1 = 35;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    cellpoissondiag2d_(i0, j0, i1, j1, sgcw, st.host_data());

    coeffs<T>::poisson_diag(
        i0, j0, i1, j1, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adjdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto b0 = make_md_vec<T>(j, i + 1);
    auto b1 = make_md_vec<T>(i, j + 1);

    b0.fill_random();
    b1.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);

    adjcelldiffusiondiag2d_(i0,
                            j0,
                            i1,
                            j1,
                            pi0,
                            pj0,
                            pi1,
                            pj1,
                            dir,
                            side,
                            btype,
                            exOrder,
                            dx.data(),
                            beta,
                            b0.host_data(),
                            b1.host_data(),
                            sgcw,
                            st.host_data());

    coeffs<T>::adj_diag(i0,
                        j0,
                        i1,
                        j1,
                        pi0,
                        pj0,
                        pi1,
                        pj1,
                        dir,
                        side,
                        btype,
                        exOrder,
                        dx.data(),
                        beta,
                        b0.data(),
                        b1.data(),
                        sgcw,
                        thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson adjdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissondiag2d_(i0,
                          j0,
                          i1,
                          j1,
                          pi0,
                          pj0,
                          pi1,
                          pj1,
                          dir,
                          side,
                          btype,
                          exOrder,
                          dx.data(),
                          beta,
                          sgcw,
                          st.host_data());

    coeffs<T>::adj_poisson_diag(i0,
                                j0,
                                i1,
                                j1,
                                pi0,
                                pj0,
                                pi1,
                                pj1,
                                dir,
                                side,
                                btype,
                                exOrder,
                                dx.data(),
                                beta,
                                sgcw,
                                thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj cfdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto b0 = make_md_vec<T>(j, i + 1);
    auto b1 = make_md_vec<T>(i, j + 1);

    b0.fill_random();
    b1.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcelldiffusioncfdiag2d_(i0,
                              j0,
                              i1,
                              j1,
                              pi0,
                              pj0,
                              pi1,
                              pj1,
                              r,
                              dir,
                              side,
                              intOrder,
                              dx.data(),
                              beta,
                              b0.host_data(),
                              b1.host_data(),
                              sgcw,
                              st.host_data());

    coeffs<T>::adj_cf_diag(i0,
                           j0,
                           i1,
                           j1,
                           pi0,
                           pj0,
                           pi1,
                           pj1,
                           r,
                           dir,
                           side,
                           intOrder,
                           dx.data(),
                           beta,
                           b0.data(),
                           b1.data(),
                           sgcw,
                           thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson cfdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcellpoissoncfdiag2d_(i0,
                            j0,
                            i1,
                            j1,
                            pi0,
                            pj0,
                            pi1,
                            pj1,
                            r,
                            dir,
                            side,
                            intOrder,
                            dx.data(),
                            beta,
                            sgcw,
                            st.host_data());

    coeffs<T>::adj_poisson_cf_diag(i0,
                                   j0,
                                   i1,
                                   j1,
                                   pi0,
                                   pj0,
                                   pi1,
                                   pj1,
                                   r,
                                   dir,
                                   side,
                                   intOrder,
                                   dx.data(),
                                   beta,
                                   sgcw,
                                   thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick(0.1);
    const T neu_factor = pick(0.1);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto b0 = make_md_vec<T>(j, i + 1);
    auto b1 = make_md_vec<T>(i, j + 1);

    b0.fill_random();
    b1.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);
    adjcelldiffusionoffdiag2d_(i0,
                               j0,
                               i1,
                               j1,
                               pi0,
                               pj0,
                               pi1,
                               pj1,
                               dir,
                               side,
                               btype,
                               exOrder,
                               dx.data(),
                               dir_factor,
                               neu_factor,
                               beta,
                               b0.host_data(),
                               b1.host_data(),
                               sgcw,
                               st.host_data());

    coeffs<T>::adj_offdiag(i0,
                           j0,
                           i1,
                           j1,
                           pi0,
                           pj0,
                           pi1,
                           pj1,
                           dir,
                           side,
                           btype,
                           exOrder,
                           dx.data(),
                           dir_factor,
                           neu_factor,
                           beta,
                           b0.data(),
                           b1.data(),
                           sgcw,
                           thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick(0.1);
    const T neu_factor = pick(0.1);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);
    adjcellpoissonoffdiag2d_(i0,
                             j0,
                             i1,
                             j1,
                             pi0,
                             pj0,
                             pi1,
                             pj1,
                             dir,
                             side,
                             btype,
                             exOrder,
                             dx.data(),
                             dir_factor,
                             neu_factor,
                             beta,
                             sgcw,
                             st.host_data());

    coeffs<T>::adj_poisson_offdiag(i0,
                                   j0,
                                   i1,
                                   j1,
                                   pi0,
                                   pj0,
                                   pi1,
                                   pj1,
                                   dir,
                                   side,
                                   btype,
                                   exOrder,
                                   dx.data(),
                                   dir_factor,
                                   neu_factor,
                                   beta,
                                   sgcw,
                                   thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj cf offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcelldiffusioncfoffdiag2d_(
        i0, j0, i1, j1, pi0, pj0, pi1, pj1, r, dir, side, intOrder, sgcw, st.host_data());

    coeffs<T>::adj_cf_offdiag(i0,
                              j0,
                              i1,
                              j1,
                              pi0,
                              pj0,
                              pi1,
                              pj1,
                              r,
                              dir,
                              side,
                              intOrder,
                              sgcw,
                              thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("readj offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);
    readjcelldiffusionoffdiag2d_(
        i0, j0, i1, j1, pi0, pj0, pi1, pj1, dir, side, sgcw, st.host_data());

    coeffs<T>::readj_offdiag(i0,
                             j0,
                             i1,
                             j1,
                             pi0,
                             pj0,
                             pi1,
                             pj1,
                             dir,
                             side,
                             sgcw,
                             thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj cf bdryrhs")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90;
    const int i1 = 31, j1 = 131;
    const int pi0 = i0 + 1, pj0 = j0 + 2;
    const int pi1 = i1 - 3, pj1 = j1 - 4;
    const int sgcw = pick(1, 3);
    const int gcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto u = make_md_vec<T>(gcw, j, i);
    auto rhs = make_md_vec<T>(j, i);

    u.fill_random();
    rhs.fill_random();
    st.fill_random();

    thrust::device_vector<T> rhs_cuda = rhs.host();

    auto dir = GENERATE(0, 1);
    auto side = GENERATE(0, 1);

    adjcelldiffusioncfbdryrhs2d_(i0,
                                 j0,
                                 i1,
                                 j1,
                                 pi0,
                                 pj0,
                                 pi1,
                                 pj1,
                                 dir,
                                 side,
                                 sgcw,
                                 st.host_data(),
                                 gcw,
                                 u.host_data(),
                                 rhs.host_data());

    coeffs<T>::adj_cf_bdryrhs(i0,
                              j0,
                              i1,
                              j1,
                              pi0,
                              pj0,
                              pi1,
                              pj1,
                              dir,
                              side,
                              sgcw,
                              st.data(),
                              gcw,
                              u.data(),
                              thrust::raw_pointer_cast(rhs_cuda.data()));

    compare<T>(rhs, rhs_cuda);
}
