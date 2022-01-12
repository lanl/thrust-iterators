#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_stencil_coeffs_1d_cuda.hpp"

/*
 * Args to celldiffusionoffdiag1d: ( I think the signature is off...)

 *  int&: ifirst0,ilast0 <- stencil(0:2, ifirst0-sgcw:ilast0+sgcw)
 *  int&: bilo0, bihi0 <-   b0(bilo0:bihi0+1)
 *  double* : dx(0:0)
 *  double& : beta
 *  double* : b0(bilo0:bihi0+1)
 *  int&: sgcw
 *  double* : stencil(0:2, ifirst0-sgcw:ilast0+sgcw)
 *
 * routine sets stencil(1:2,ifirst0:ilast0) using b0(ifirst0:ilast0+1)
 * -> bilo0 <= ifirst0 and bihi0 >= ilast0

*/

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

TEST_CASE("offdiag")
{
    const int ifirst = 1;
    const int ilast = 11;
    const int bilo0 = 0;
    const int bihi0 = 20;
    const std::array dx{0.5};
    const double beta = 2.0;
    const int sgcw = 1;
    std::vector<double> b0(2 + bihi0 - bilo0);
    std::vector<double> st(3 * (1 + ilast + sgcw - (ifirst - sgcw)));

    randomize();
    std::generate(b0.begin(), b0.end(), f);
    celldiffusionoffdiag1d_(
        ifirst, ilast, bilo0, bihi0, &dx[0], beta, &b0[0], sgcw, &st[0]);

    std::vector<double> st_cuda(st.size());
    cd_stencil_coeffs_1d_cuda<>::offdiag(
        ifirst, ilast, bilo0, bihi0, &dx[0], beta, &b0[0], sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}

TEST_CASE("poisson offdiag")
{
    const int ifirst = 1;
    const int ilast = 11;
    const std::array dx{0.5};
    const double beta = 2.0;
    const int sgcw = 1;
    std::vector<double> st(3 * (1 + ilast + sgcw - (ifirst - sgcw)), -1);

    randomize();
    cellpoissonoffdiag1d_(ifirst, ilast, &dx[0], beta, sgcw, &st[0]);

    std::vector<double> st_cuda(st.size(), -1);
    cd_stencil_coeffs_1d_cuda<>::poisson_offdiag(
        ifirst, ilast, &dx[0], beta, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}

TEST_CASE("v1diag")
{
    const int ifirst = 1;
    const int ilast = 11;
    const int ailo0 = -1;
    const int aihi0 = 20;
    const std::array dx{0.5};
    const double alpha = 2.0;
    const int sgcw = 1;
    std::vector<double> a(1 + aihi0 - ailo0);
    std::vector<double> st(3 * (1 + ilast + sgcw - (ifirst - sgcw)));

    randomize();

    std::generate(a.begin(), a.end(), f);
    std::generate(st.begin(), st.end(), f);
    std::vector<double> st_cuda{st};

    celldiffusionv1diag1d_(ifirst, ilast, ailo0, aihi0, alpha, &a[0], sgcw, &st[0]);

    cd_stencil_coeffs_1d_cuda<>::v1diag(
        ifirst, ilast, ailo0, aihi0, alpha, &a[0], sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}

TEST_CASE("v2diag")
{
    const int ifirst = 1;
    const int ilast = 11;
    const double alpha = 2.0;
    const int sgcw = 1;
    std::vector<double> st(3 * (1 + ilast + sgcw - (ifirst - sgcw)));

    randomize();

    std::generate(st.begin(), st.end(), f);
    std::vector<double> st_cuda{st};

    celldiffusionv2diag1d_(ifirst, ilast, alpha, sgcw, &st[0]);

    cd_stencil_coeffs_1d_cuda<>::v2diag(ifirst, ilast, alpha, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}


TEST_CASE("poisson diag")
{
    const int ifirst = 1;
    const int ilast = 11;
    const int sgcw = 1;
    std::vector<double> st(3 * (1 + ilast + sgcw - (ifirst - sgcw)));

    randomize();

    std::generate(st.begin(), st.end(), f);
    std::vector<double> st_cuda{st};

    cellpoissondiag1d_(ifirst, ilast, sgcw, &st[0]);

    cd_stencil_coeffs_1d_cuda<>::poisson_diag(ifirst, ilast, sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}
