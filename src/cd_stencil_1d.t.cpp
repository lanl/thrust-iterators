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
#include "md_device_vector.hpp"

using Catch::Matchers::Approx;

template <typename T>
using coeffs = cd_stencil_1d_cuda<T>;

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

constexpr auto w = Wb{0, 2};

TEST_CASE("offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int bi0 = i0 - pick(1, 3);
    const int bi1 = i1 + pick(2, 4);

    const std::array dx{0.5};
    const T beta = 2.0;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto bi = Ib{bi0, bi1};

    auto b0 = make_md_vec<T>(bi + 1);
    auto st = make_md_vec<T>(sgcw, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, i, w);

    b0.fill_random();

    celldiffusionoffdiag1d_(
        i0, i1, bi0, bi1, dx.data(), beta, b0.host_data(), sgcw, st.host_data());

    coeffs<T>::offdiag(
        i0, i1, bi0, bi1, dx.data(), beta, b0.data(), sgcw, st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 21;
    const std::array dx{0.5};
    const T beta = 2.0;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);
    auto st_cuda = make_md_vec<T>(sgcw, i, w);

    cellpoissonoffdiag1d_(i0, i1, dx.data(), beta, sgcw, st.host_data());

    coeffs<T>::poisson_offdiag(i0, i1, dx.data(), beta, sgcw, st_cuda.data());

    compare<T>(st, st_cuda);
}

TEST_CASE("v1diag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int ai0 = i0 - 2;
    const int ai1 = i1 + 3;
    const T alpha = pick(0.1, 10.0);

    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    const auto ai = Ib{ai0, ai1};

    auto a = make_md_vec<T>(ai);
    auto st = make_md_vec<T>(sgcw, i, w);

    a.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv1diag1d_(i0, i1, ai0, ai1, alpha, a.host_data(), sgcw, st.host_data());

    coeffs<T>::v1diag(i0,
                      i1,
                      ai0,
                      ai1,
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

    const int i0 = 11;
    const int i1 = 26;
    const int sgcw = pick(1, 3);
    const T alpha = pick(0.1, 10.0);

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    celldiffusionv2diag1d_(i0, i1, alpha, sgcw, st.host_data());

    coeffs<T>::v2diag(i0, i1, alpha, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson diag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 21;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};
    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    cellpoissondiag1d_(i0, i1, sgcw, st.host_data());

    coeffs<T>::poisson_diag(i0, i1, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adjdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1)};

    const auto i = Ib{i0, i1};

    auto b0 = make_md_vec<T>(i + 1);
    auto st = make_md_vec<T>(sgcw, i, w);

    b0.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

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
                            b0.host_data(),
                            sgcw,
                            st.host_data());

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
                        thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("poisson adjdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 2> dx{pick(0.1)};

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    int dir = 0;
    auto side = GENERATE(0, 1);
    auto btype = GENERATE(0, 4);
    auto exOrder = GENERATE(1, 2);

    adjcellpoissondiag1d_(
        i0, i1, pi0, pi1, side, btype, exOrder, dx.data(), beta, sgcw, st.host_data());

    coeffs<T>::adj_poisson_diag(i0,
                                i1,
                                pi0,
                                pi1,
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

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};

    auto b0 = make_md_vec<T>(i + 1);
    auto st = make_md_vec<T>(sgcw, i, Wb{0, 6});

    b0.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcelldiffusioncfdiag1d_(i0,
                              i1,
                              pi0,
                              pi1,
                              r,
                              side,
                              intOrder,
                              dx.data(),
                              beta,
                              b0.host_data(),
                              sgcw,
                              st.host_data());

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
                           thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson cfdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    std::array<T, 3> dx{pick(0.1), pick(0.1), pick(0.1)};

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, Wb{0, 6});

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);
    adjcellpoissoncfdiag1d_(
        i0, i1, pi0, pi1, r, side, intOrder, dx.data(), beta, sgcw, st.host_data());

    coeffs<T>::adj_poisson_cf_diag(i0,
                                   i1,
                                   pi0,
                                   pi1,
                                   r,
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

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const T beta = pick(0.1, 10.0);
    const T dir_factor = pick(0.1);
    const T neu_factor = pick(0.1);

    std::array<T, 2> dx{pick(0.1)};

    const auto i = Ib{i0, i1};

    auto b0 = make_md_vec<T>(i + 1);
    auto st = make_md_vec<T>(sgcw, i, w);

    b0.fill_random();
    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

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
                               b0.host_data(),
                               sgcw,
                               st.host_data());

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
                           thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj poisson offdiag")
{
    using T = double;
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

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

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
                             st.host_data());

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
                                   thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj cf offdiag")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto side = GENERATE(0, 1);
    auto r = GENERATE(1, 4);
    auto intOrder = GENERATE(1, 2);

    adjcelldiffusioncfoffdiag1d_(
        i0, i1, pi0, pi1, r, side, intOrder, sgcw, st.host_data());

    coeffs<T>::adj_cf_offdiag(i0,
                              i1,
                              pi0,
                              pi1,
                              r,
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

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 1;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);

    st.fill_random();

    thrust::device_vector<T> st_cuda = st.host();

    auto side = GENERATE(0, 1);

    readjcelldiffusionoffdiag1d_(i0, i1, pi0, pi1, side, sgcw, st.host_data());

    coeffs<T>::readj_offdiag(
        i0, i1, pi0, pi1, side, sgcw, thrust::raw_pointer_cast(st_cuda.data()));

    compare<T>(st, st_cuda);
}

TEST_CASE("adj cf bdryrhs")
{
    using T = double;
    randomize();

    const int i0 = 10;
    const int i1 = 31;
    const int pi0 = i0 + 4;
    const int pi1 = i1 - 3;
    const int sgcw = pick(1, 3);
    const int gcw = pick(1, 3);

    const auto i = Ib{i0, i1};

    auto st = make_md_vec<T>(sgcw, i, w);
    auto u = make_md_vec<T>(gcw, i);
    auto rhs = make_md_vec<T>(i);

    u.fill_random();
    rhs.fill_random();
    st.fill_random();

    thrust::device_vector<T> rhs_cuda = rhs.host();

    auto side = GENERATE(0, 1);

    adjcelldiffusioncfbdryrhs1d_(i0,
                                 i1,
                                 pi0,
                                 pi1,
                                 side,
                                 sgcw,
                                 st.host_data(),
                                 gcw,
                                 u.host_data(),
                                 rhs.host_data());

    coeffs<T>::adj_cf_bdryrhs(i0,
                              i1,
                              pi0,
                              pi1,
                              side,
                              sgcw,
                              st.data(),
                              gcw,
                              u.data(),
                              thrust::raw_pointer_cast(rhs_cuda.data()));

    compare<T>(rhs, rhs_cuda);
}
