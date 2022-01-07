#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

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

TEST_CASE("stencil")
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
    cd_stencil_coeffs_1d_cuda<>::offdiag1d(
        ifirst, ilast, bilo0, bihi0, &dx[0], beta, &b0[0], sgcw, &st_cuda[0]);

    REQUIRE_THAT(st, Approx(st_cuda));
}
