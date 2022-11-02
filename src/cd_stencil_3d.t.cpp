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


#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_stencil_3d_cuda.hpp"
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

constexpr auto w = Wb{0, 6};

TEST_CASE("offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;
    const int bi0 = i0 - 2, bj0 = j0 - 3, bk0 = k0 - 1;
    const int bi1 = i1 + 1, bj1 = j1 + 2, bk1 = k1 + 3;

    const std::array dx{pick(0.1), pick(0.1), pick(0.1)};
    const T beta = pick(0.2, 10.0);
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1}, bi = Ib{bi0, bi1};
    const auto j = Jb{j0, j1}, bj = Jb{bj0, bj1};
    const auto k = Kb{k0, k1}, bk = Kb{bk0, bk1};

    auto b0 = make_md_vec<T>(bk, bj, bi + 1);
    auto b1 = make_md_vec<T>(bi, bk, bj + 1);
    auto b2 = make_md_vec<T>(bj, bi, bk + 1);
    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, k, j, i, w);

    b0.fill_random();
    b1.fill_random();
    b2.fill_random();

    celldiffusionoffdiag3d_(i0,
                            j0,
                            k0,
                            i1,
                            j1,
                            k1,
                            bi0,
                            bj0,
                            bk0,
                            bi1,
                            bj1,
                            bk1,
                            dx.data(),
                            beta,
                            b0.host_data(),
                            b1.host_data(),
                            b2.host_data(),
                            sgcw,
                            st.host_data());

    cd_stencil_3d_cuda<T>::offdiag(i0,
                                   j0,
                                   k0,
                                   i1,
                                   j1,
                                   k1,
                                   bi0,
                                   bj0,
                                   bk0,
                                   bi1,
                                   bj1,
                                   bk1,
                                   dx.data(),
                                   beta,
                                   b0.data(),
                                   b1.data(),
                                   b2.data(),
                                   sgcw,
                                   st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;

    const std::array dx{pick(0.1), pick(0.1), pick(0.1)};
    const T beta = pick(0.2, 10.0);
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, k, j, i, w);

    cellpoissonoffdiag3d_(i0, j0, k0, i1, j1, k1, dx.data(), beta, sgcw, st.host_data());

    cd_stencil_3d_cuda<T>::poisson_offdiag(
        i0, j0, k0, i1, j1, k1, dx.data(), beta, sgcw, st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("v1diag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int ai0 = i0 - 1, aj0 = j0 - 2, ak0 = k0 - 3;
    const int ai1 = i1 + 3, aj1 = j1 + 1, ak1 = k1 + 2;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    const auto i = Ib{i0, i1}, ai = Ib{ai0, ai1};
    const auto j = Jb{j0, j1}, aj = Jb{aj0, aj1};
    const auto k = Kb{k0, k1}, ak = Kb{ak0, ak1};

    auto a = make_md_vec<T>(ak, aj, ai);
    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    a.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv1diag3d_(i0,
                           j0,
                           k0,
                           i1,
                           j1,
                           k1,
                           ai0,
                           aj0,
                           ak0,
                           ai1,
                           aj1,
                           ak1,
                           alpha,
                           a.host_data(),
                           sgcw,
                           st.host_data());

    cd_stencil_3d_cuda<T>::v1diag(i0,
                                  j0,
                                  k0,
                                  i1,
                                  j1,
                                  k1,
                                  ai0,
                                  aj0,
                                  ak0,
                                  ai1,
                                  aj1,
                                  ak1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv2diag3d_(i0, j0, k0, i1, j1, k1, alpha, sgcw, st.host_data());

    cd_stencil_3d_cuda<T>::v2diag(
        i0, j0, k0, i1, j1, k1, alpha, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson diag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;

    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    cellpoissondiag3d_(i0, j0, k0, i1, j1, k1, sgcw, st.host_data());

    cd_stencil_3d_cuda<T>::poisson_diag(
        i0, j0, k0, i1, j1, k1, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adjdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto b0 = make_md_vec<T>(k, j, i + 1);
    auto b1 = make_md_vec<T>(i, k, j + 1);
    auto b2 = make_md_vec<T>(j, i, k + 1);

    b0.fill_random();
    b1.fill_random();
    b2.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);
    adjcelldiffusiondiag3d_(i0,
                            j0,
                            k0,
                            i1,
                            j1,
                            k1,
                            pi0,
                            pj0,
                            pk0,
                            pi1,
                            pj1,
                            pk1,
                            dir,
                            side,
                            btype,
                            exOrder,
                            dx.data(),
                            beta,
                            b0.host_data(),
                            b1.host_data(),
                            b2.host_data(),
                            sgcw,
                            st.host_data());

    cd_stencil_3d_cuda<T>::adj_diag(i0,
                                    j0,
                                    k0,
                                    i1,
                                    j1,
                                    k1,
                                    pi0,
                                    pj0,
                                    pk0,
                                    pi1,
                                    pj1,
                                    pk1,
                                    dir,
                                    side,
                                    btype,
                                    exOrder,
                                    dx.data(),
                                    beta,
                                    b0.data(),
                                    b1.data(),
                                    b2.data(),
                                    sgcw,
                                    thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj_poissondiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissondiag3d_(i0,
                          j0,
                          k0,
                          i1,
                          j1,
                          k1,
                          pi0,
                          pj0,
                          pk0,
                          pi1,
                          pj1,
                          pk1,
                          dir,
                          side,
                          btype,
                          exOrder,
                          dx.data(),
                          beta,
                          sgcw,
                          st.host_data());

    cd_stencil_3d_cuda<T>::adj_poisson_diag(i0,
                                            j0,
                                            k0,
                                            i1,
                                            j1,
                                            k1,
                                            pi0,
                                            pj0,
                                            pk0,
                                            pi1,
                                            pj1,
                                            pk1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto b0 = make_md_vec<T>(k, j, i + 1);
    auto b1 = make_md_vec<T>(i, k, j + 1);
    auto b2 = make_md_vec<T>(j, i, k + 1);

    b0.fill_random();
    b1.fill_random();
    b2.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcelldiffusioncfdiag3d_(i0,
                              j0,
                              k0,
                              i1,
                              j1,
                              k1,
                              pi0,
                              pj0,
                              pk0,
                              pi1,
                              pj1,
                              pk1,
                              r,
                              dir,
                              side,
                              intOrder,
                              dx.data(),
                              beta,
                              b0.host_data(),
                              b1.host_data(),
                              b2.host_data(),
                              sgcw,
                              st.host_data());

    cd_stencil_3d_cuda<T>::adj_cf_diag(i0,
                                       j0,
                                       k0,
                                       i1,
                                       j1,
                                       k1,
                                       pi0,
                                       pj0,
                                       pk0,
                                       pi1,
                                       pj1,
                                       pk1,
                                       r,
                                       dir,
                                       side,
                                       intOrder,
                                       dx.data(),
                                       beta,
                                       b0.data(),
                                       b1.data(),
                                       b2.data(),
                                       sgcw,
                                       thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson cfdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcellpoissoncfdiag3d_(i0,
                            j0,
                            k0,
                            i1,
                            j1,
                            k1,
                            pi0,
                            pj0,
                            pk0,
                            pi1,
                            pj1,
                            pk1,
                            r,
                            dir,
                            side,
                            intOrder,
                            dx.data(),
                            beta,
                            sgcw,
                            st.host_data());

    cd_stencil_3d_cuda<T>::adj_poisson_cf_diag(i0,
                                               j0,
                                               k0,
                                               i1,
                                               j1,
                                               k1,
                                               pi0,
                                               pj0,
                                               pk0,
                                               pi1,
                                               pj1,
                                               pk1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick();
    const T neu_factor = pick();
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto b0 = make_md_vec<T>(k, j, i + 1);
    auto b1 = make_md_vec<T>(i, k, j + 1);
    auto b2 = make_md_vec<T>(j, i, k + 1);

    b0.fill_random();
    b1.fill_random();
    b2.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);

    adjcelldiffusionoffdiag3d_(i0,
                               j0,
                               k0,
                               i1,
                               j1,
                               k1,
                               pi0,
                               pj0,
                               pk0,
                               pi1,
                               pj1,
                               pk1,
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
                               b2.host_data(),
                               sgcw,
                               st.host_data());

    cd_stencil_3d_cuda<T>::adj_offdiag(i0,
                                       j0,
                                       k0,
                                       i1,
                                       j1,
                                       k1,
                                       pi0,
                                       pj0,
                                       pk0,
                                       pi1,
                                       pj1,
                                       pk1,
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
                                       b2.data(),
                                       sgcw,
                                       thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick();
    const T neu_factor = pick();
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissonoffdiag3d_(i0,
                             j0,
                             k0,
                             i1,
                             j1,
                             k1,
                             pi0,
                             pj0,
                             pk0,
                             pi1,
                             pj1,
                             pk1,
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

    cd_stencil_3d_cuda<T>::adj_poisson_offdiag(i0,
                                               j0,
                                               k0,
                                               i1,
                                               j1,
                                               k1,
                                               pi0,
                                               pj0,
                                               pk0,
                                               pi1,
                                               pj1,
                                               pk1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);

    adjcelldiffusioncfoffdiag3d_(i0,
                                 j0,
                                 k0,
                                 i1,
                                 j1,
                                 k1,
                                 pi0,
                                 pj0,
                                 pk0,
                                 pi1,
                                 pj1,
                                 pk1,
                                 r,
                                 dir,
                                 side,
                                 intOrder,
                                 sgcw,
                                 st.host_data());

    cd_stencil_3d_cuda<T>::adj_cf_offdiag(i0,
                                          j0,
                                          k0,
                                          i1,
                                          j1,
                                          k1,
                                          pi0,
                                          pj0,
                                          pk0,
                                          pi1,
                                          pj1,
                                          pk1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);

    readjcelldiffusionoffdiag3d_(i0,
                                 j0,
                                 k0,
                                 i1,
                                 j1,
                                 k1,
                                 pi0,
                                 pj0,
                                 pk0,
                                 pi1,
                                 pj1,
                                 pk1,
                                 dir,
                                 side,
                                 sgcw,
                                 st.host_data());

    cd_stencil_3d_cuda<T>::readj_offdiag(i0,
                                         j0,
                                         k0,
                                         i1,
                                         j1,
                                         k1,
                                         pi0,
                                         pj0,
                                         pk0,
                                         pi1,
                                         pj1,
                                         pk1,
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

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const int gcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto u = make_md_vec<T>(gcw, k, j, i);
    auto rhs = make_md_vec<T>(k, j, i);

    st.fill_random();
    u.fill_random();
    rhs.fill_random();

    thrust::device_vector<T> rhs_cuda = rhs.host();

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);

    adjcelldiffusioncfbdryrhs3d_(i0,
                                 j0,
                                 k0,
                                 i1,
                                 j1,
                                 k1,
                                 pi0,
                                 pj0,
                                 pk0,
                                 pi1,
                                 pj1,
                                 pk1,
                                 dir,
                                 side,
                                 sgcw,
                                 st.host_data(),
                                 gcw,
                                 u.host_data(),
                                 rhs.host_data());

    cd_stencil_3d_cuda<T>::adj_cf_bdryrhs(i0,
                                          j0,
                                          k0,
                                          i1,
                                          j1,
                                          k1,
                                          pi0,
                                          pj0,
                                          pk0,
                                          pi1,
                                          pj1,
                                          pk1,
                                          dir,
                                          side,
                                          sgcw,
                                          st.data(),
                                          gcw,
                                          u.data(),
                                          thrust::raw_pointer_cast(rhs_cuda.data()));

    compare<T>(rhs, rhs_cuda);
}
