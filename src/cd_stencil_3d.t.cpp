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
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;

TEST_CASE("offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;
    const int bi0 = i0 - 2, bj0 = j0 - 3, bk0 = k0 - 1;
    const int bi1 = i1 + 1, bj1 = j1 + 2, bk1 = k1 + 3;

    const std::array dx{pick(0.1), pick(0.1), pick(0.1)};
    const T beta = pick(0.2, 10.0);
    const int sgcw = pick(1, 3);

    vec b0(B{bk0, bk1}, B{bj0, bj1}, B{bi0, bi1 + 1});
    vec b1(B{bi0, bi1}, B{bk0, bk1}, B{bj0, bj1 + 1});
    vec b2(B{bj0, bj1}, B{bi0, bi1}, B{bk0, bk1 + 1});
    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(b2.begin(), b2.end(), f);
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
                            b0.data(),
                            b1.data(),
                            b2.data(),
                            sgcw,
                            st.data());

    std::vector<T> st_cuda(st.size());
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
                                   &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("poisson offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;

    const std::array dx{pick(0.1), pick(0.1), pick(0.1)};
    const T beta = pick(0.2, 10.0);
    const int sgcw = pick(1, 3);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    cellpoissonoffdiag3d_(i0, j0, k0, i1, j1, k1, dx.data(), beta, sgcw, st.data());

    std::vector<T> st_cuda(st.size());
    cd_stencil_3d_cuda<T>::poisson_offdiag(
        i0, j0, k0, i1, j1, k1, dx.data(), beta, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("v1diag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int ai0 = i0 - 1, aj0 = j0 - 2, ak0 = k0 - 3;
    const int ai1 = i1 + 3, aj1 = j1 + 1, ak1 = k1 + 2;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});
    vec a{B{ak0, ak1}, B{aj0, aj1}, B{ai0, ai1}};

    std::generate(st.begin(), st.end(), f);
    std::generate(a.begin(), a.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                           a.data(),
                           sgcw,
                           st.data());

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
                                  &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("v2diag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    celldiffusionv2diag3d_(i0, j0, k0, i1, j1, k1, alpha, sgcw, st.data());

    cd_stencil_3d_cuda<T>::v2diag(i0, j0, k0, i1, j1, k1, alpha, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("poisson diag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 40, k0 = 90;
    const int i1 = 31, j1 = 78, k1 = 131;

    const int sgcw = pick(1, 3);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    cellpoissondiag3d_(i0, j0, k0, i1, j1, k1, sgcw, st.data());

    cd_stencil_3d_cuda<T>::poisson_diag(i0, j0, k0, i1, j1, k1, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adjdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});
    vec b0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec b1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec b2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(b2.begin(), b2.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                            b0.data(),
                            b1.data(),
                            b2.data(),
                            sgcw,
                            st.data());

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
                                    &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj_poissondiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                          st.data());

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
                                            &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj cfdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});
    vec b0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec b1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec b2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(b2.begin(), b2.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                              b0.data(),
                              b1.data(),
                              b2.data(),
                              sgcw,
                              st.data());

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
                                       &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj poisson cfdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 131, k1 = 78;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                            st.data());

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
                                               &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
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

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});
    vec b0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec b1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec b2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);
    std::generate(b1.begin(), b1.end(), f);
    std::generate(b2.begin(), b2.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                               b0.data(),
                               b1.data(),
                               b2.data(),
                               sgcw,
                               st.data());

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
                                       &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj poisson offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
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

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                             st.data());

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
                                               &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj cf offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

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
                                 st.data());

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
                                          &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("readj offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    auto dir = GENERATE(0, 1, 2);
    auto side = GENERATE(0, 1);

    readjcelldiffusionoffdiag3d_(
        i0, j0, k0, i1, j1, k1, pi0, pj0, pk0, pi1, pj1, pk1, dir, side, sgcw, st.data());

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
                                         &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj cf bdryrhs")
{
    using T = double;
    using vec = md_host_vector<T, 3>;
    using wvec = md_host_vector<T, 4>;
    randomize();

    const int i0 = 10, j0 = 90, k0 = 40;
    const int i1 = 31, j1 = 111, k1 = 55;
    const int pi0 = i0 + 1, pj0 = j0 + 2, pk0 = k0 + 3;
    const int pi1 = i1 - 3, pj1 = j1 - 4, pk1 = k1 - 4;
    const int sgcw = pick(1, 3);
    const int gcw = pick(1, 3);

    wvec st(B{k0 - sgcw, k1 + sgcw},
            B{j0 - sgcw, j1 + sgcw},
            B{i0 - sgcw, i1 + sgcw},
            B{0, 6});
    vec u(B{k0 - gcw, k1 + gcw}, B{j0 - gcw, j1 + gcw}, B{i0 - gcw, i1 + gcw});
    vec rhs(B{k0, k1}, B{j0, j1}, B{i0, i1});

    std::generate(st.begin(), st.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::generate(rhs.begin(), rhs.end(), f);
    std::vector<T> rhs_cuda{rhs.vec()};

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
                                 st.data(),
                                 gcw,
                                 u.data(),
                                 rhs.data());

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
                                          &rhs_cuda[0]);

    REQUIRE_THAT(rhs.vec(), Approx(rhs_cuda));
}
