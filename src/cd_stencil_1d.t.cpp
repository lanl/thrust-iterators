#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_stencil_1d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using coeffs = cd_stencil_1d_cuda<T>;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;

TEST_CASE("offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int bi0 = i0 - pick(1, 3);
    const int bi1 = i1 + pick(2, 4);

    const std::array dx{0.5};
    const T beta = 2.0;
    const int sgcw = pick(1, 3);

    vec b0(B{bi0, bi1 + 1});
    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    randomize();
    std::generate(b0.begin(), b0.end(), f);
    celldiffusionoffdiag1d_(
        i0, i1, bi0, bi1, dx.data(), beta, b0.data(), sgcw, st.data());

    std::vector<T> st_cuda(st.size());
    coeffs<T>::offdiag(i0, i1, bi0, bi1, dx.data(), beta, b0.data(), sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("poisson offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 21;
    const std::array dx{0.5};
    const T beta = 2.0;
    const int sgcw = pick(1, 3);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    cellpoissonoffdiag1d_(i0, i1, dx.data(), beta, sgcw, st.data());

    std::vector<T> st_cuda(st.size());
    coeffs<T>::poisson_offdiag(i0, i1, dx.data(), beta, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("v1diag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int ai0 = i0 - 2;
    const int ai1 = i1 + 3;
    const T alpha = pick(0.1, 10.0);

    const int sgcw = pick(1, 3);

    vec a(B{ai0, ai1});
    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    std::generate(a.begin(), a.end(), f);
    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    celldiffusionv1diag1d_(i0, i1, ai0, ai1, alpha, a.data(), sgcw, st.data());

    coeffs<T>::v1diag(i0, i1, ai0, ai1, alpha, a.data(), sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("v2diag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 11;
    const int i1 = 26;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    // prepare data
    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    celldiffusionv2diag1d_(i0, i1, alpha, sgcw, st.data());

    coeffs<T>::v2diag(i0, i1, alpha, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("poisson diag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 21;
    const int sgcw = pick(1, 3);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    // prepare data
    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    cellpoissondiag1d_(i0, i1, sgcw, st.data());

    coeffs<T>::poisson_diag(i0, i1, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adjdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});
    vec b0(B{i0, i1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);

    std::vector<T> st_cuda{st.vec()};

    int dir = 0;
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);

    adjcelldiffusiondiag1d_(i0,
                            i1,
                            pi0,
                            pi1,
                            side,
                            btype,
                            exOrder,
                            dx.data(),
                            beta,
                            b0.data(),
                            sgcw,
                            st.data());

    coeffs<T>::adj_diag(i0,
                        i1,
                        pi0,
                        pi1,
                        side,
                        btype,
                        exOrder,
                        dx.data(),
                        beta,
                        b0.data(),
                        sgcw,
                        &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("poisson adjdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    std::generate(st.begin(), st.end(), f);

    std::vector<T> st_cuda{st.vec()};

    int dir = 0;
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissondiag1d_(
        i0, i1, pi0, pi1, side, btype, exOrder, dx.data(), beta, sgcw, st.data());

    coeffs<T>::adj_poisson_diag(
        i0, i1, pi0, pi1, side, btype, exOrder, dx.data(), beta, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj cfdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 6});
    vec b0(B{i0, i1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);
    std::vector<T> st_cuda{st.vec()};

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcelldiffusioncfdiag1d_(
        i0, i1, pi0, pi1, r, side, intOrder, dx.data(), beta, b0.data(), sgcw, st.data());

    coeffs<T>::adj_cf_diag(i0,
                           i1,
                           pi0,
                           pi1,
                           r,
                           side,
                           intOrder,
                           dx.data(),
                           beta,
                           b0.data(),
                           sgcw,
                           &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj poisson cfdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 6});

    std::generate(st.begin(), st.end(), f);
    std::vector<T> st_cuda{st.vec()};

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcellpoissoncfdiag1d_(
        i0, i1, pi0, pi1, r, side, intOrder, dx.data(), beta, sgcw, st.data());

    coeffs<T>::adj_poisson_cf_diag(
        i0, i1, pi0, pi1, r, side, intOrder, dx.data(), beta, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick(0.1);
    const T neu_factor = pick(0.1);

    std::array<T, 2> dx{pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});
    vec b0(B{i0, i1 + 1});

    std::generate(st.begin(), st.end(), f);
    std::generate(b0.begin(), b0.end(), f);

    std::vector<T> st_cuda{st.vec()};

    int dir = 0;
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);

    adjcelldiffusionoffdiag1d_(i0,
                               i1,
                               pi0,
                               pi1,
                               side,
                               btype,
                               exOrder,
                               dx.data(),
                               dir_factor,
                               neu_factor,
                               beta,
                               b0.data(),
                               sgcw,
                               st.data());

    coeffs<T>::adj_offdiag(i0,
                           i1,
                           pi0,
                           pi1,
                           side,
                           btype,
                           exOrder,
                           dx.data(),
                           dir_factor,
                           neu_factor,
                           beta,
                           b0.data(),
                           sgcw,
                           &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj poisson offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick(0.1);
    const T neu_factor = pick(0.1);

    std::array<T, 2> dx{pick(0.1)};

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    std::generate(st.begin(), st.end(), f);

    std::vector<T> st_cuda{st.vec()};

    int dir = 0;
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 1, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissonoffdiag1d_(i0,
                             i1,
                             pi0,
                             pi1,
                             side,
                             btype,
                             exOrder,
                             dx.data(),
                             dir_factor,
                             neu_factor,
                             beta,
                             sgcw,
                             st.data());

    coeffs<T>::adj_poisson_offdiag(i0,
                                   i1,
                                   pi0,
                                   pi1,
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
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    std::generate(st.begin(), st.end(), f);

    std::vector<T> st_cuda{st.vec()};

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);

    adjcelldiffusioncfoffdiag1d_(i0, i1, pi0, pi1, r, side, intOrder, sgcw, st.data());

    coeffs<T>::adj_cf_offdiag(i0, i1, pi0, pi1, r, side, intOrder, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("readj offdiag")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});

    std::generate(st.begin(), st.end(), f);

    std::vector<T> st_cuda{st.vec()};

    auto side = GENERATE(0, 1);

    readjcelldiffusionoffdiag1d_(i0, i1, pi0, pi1, side, sgcw, st.data());

    coeffs<T>::readj_offdiag(i0, i1, pi0, pi1, side, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st.vec(), Approx(st_cuda));
}

TEST_CASE("adj cf bdryrhs")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    using wvec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 4;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const int gcw = pick(1, 3);

    wvec st(B{i0 - sgcw, i1 + sgcw}, B{0, 2});
    vec u(B{i0 - gcw, i1 + gcw});
    vec rhs(B{i0, i1});

    std::generate(st.begin(), st.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::generate(rhs.begin(), rhs.end(), f);
    std::vector<T> rhs_cuda{rhs.vec()};

    auto side = GENERATE(0, 1);

    adjcelldiffusioncfbdryrhs1d_(
        i0, i1, pi0, pi1, side, sgcw, st.data(), gcw, u.data(), rhs.data());

    coeffs<T>::adj_cf_bdryrhs(
        i0, i1, pi0, pi1, side, sgcw, st.data(), gcw, u.data(), &rhs_cuda[0]);

    REQUIRE_THAT(rhs.vec(), Approx(rhs_cuda));
}
