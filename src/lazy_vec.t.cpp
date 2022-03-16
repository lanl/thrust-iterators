#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "random.hpp"
#include <algorithm>

#include "lazy_vec.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };

using B = hbounds;
static constexpr auto N = 1;
using Dims = std::array<B, N>;

TEST_CASE("init")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 3, i1 = 8;
    const std::array dx{0.1};
    const int ugcw = pick(1, 3);
    const int rgcw = pick(ugcw, 2 * ugcw);
    const double beta = f();

    vec u(B{i0 - ugcw, i1 + ugcw});
    vec res(B{i0 - rgcw, i1 + rgcw});

    std::generate(u.begin(), u.end(), f);
    lazy_vec_cuda<T>::init(i0, i1, beta, dx.data(), ugcw, u.data(), rgcw, res.data());

    std::vector<T> expected(res.size());
    const auto& v = u.vec();

    int udims[] = {i1 + ugcw - (i0 - ugcw) + 1};
    int rdims[] = {i1 + rgcw - (i0 - rgcw) + 1};

    for (int i = i0; i <= i1; i++) {
        int ui = i - (i0 - ugcw);
        int ri = i - (i0 - rgcw);
        auto grad_down = (v[ui] - v[ui - 1]) / dx[0];
        auto grad_up = (v[ui + 1] - v[ui]) / dx[0];
        expected[ri] = 2 * (grad_down + grad_up) / (v[ui] + 10);
    }

    REQUIRE_THAT(res.vec(), Approx(expected));
}

TEST_CASE("init2")
{
    using T = double;
    using vec = md_host_vector<T, 2>;
    randomize();

    const int i0 = 3, i1 = 8, j0 = 4, j1 = 12;
    const std::array dx{0.1, 0.3};
    const int ugcw = pick(1, 3);
    const int rgcw = pick(ugcw, 2 * ugcw);
    const double beta = f();

    vec u(B{i0 - ugcw, i1 + ugcw}, B{j0 - ugcw, j1 + ugcw});
    vec res(B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});

    std::generate(u.begin(), u.end(), f);
    lazy_vec_cuda<T>::init(
        i0, i1, j0, j1, beta, dx.data(), ugcw, u.data(), rgcw, res.data());

    std::vector<T> expected(res.size());
    const auto& uv = u.vec();

    int udims[] = {j1 + ugcw - (j0 - ugcw) + 1, i1 + ugcw - (i0 - ugcw) + 1};
    int rdims[] = {i1 + rgcw - (i0 - rgcw) + 1, j1 + rgcw - (j0 - rgcw) + 1};

    for (int j = j0; j <= j1; j++)
        for (int i = i0; i <= i1; i++) {
            int ui = i - (i0 - ugcw), uj = j - (j0 - ugcw);
            int ri = i - (i0 - rgcw), rj = j - (j0 - rgcw);
            auto grad_x = (uv[(ui + 1) * udims[0] + uj] - uv[ui * udims[0] + uj]) / dx[0];
            auto grad_y = (uv[ui * udims[0] + uj + 1] - uv[ui * udims[0] + uj]) / dx[1];
            expected[rj * rdims[0] + ri] = 3 * (grad_x + grad_y);
        }
    REQUIRE_THAT(res.vec(), Approx(expected));
}

// TEST_CASE("stencil")
// {
//     using T = double;
//     using vec = md_host_vector<T, 2>;
//     randomize();

//     const int i0 = 3, i1 = 8, j0 = 4, j1 = 12;
//     const int ugcw = pick(1, 3);
//     const int rgcw = pick(ugcw, 2 * ugcw);
//     const double beta = f();

//     vec u(B{i0 - ugcw, i1 + ugcw}, B{j0 - ugcw, j1 + ugcw});
//     vec res(B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});

//     std::generate(u.begin(), u.end(), f);
//     lazy_vec_cuda<T>::stencil(i0, i1, j0, j1, beta, ugcw, u.data(), rgcw, res.data());

//     std::vector<T> expected(res.size());
//     const auto& uv = u.vec();

//     int udims[] = {j1 + ugcw - (j0 - ugcw) + 1, i1 + ugcw - (i0 - ugcw) + 1};
//     int rdims[] = {i1 + rgcw - (i0 - rgcw) + 1, j1 + rgcw - (j0 - rgcw) + 1};

//     for (int j = j0; j <= j1; j++)
//         for (int i = i0; i <= i1; i++) {
//             int ui = i - (i0 - ugcw), uj = j - (j0 - ugcw);
//             int ri = i - (i0 - rgcw), rj = j - (j0 - rgcw);
//             auto st_x = (uv[(ui + 1) * udims[0] + uj] + uv[ui * udims[0] + uj]);
//             auto st_y = (uv[ui * udims[0] + uj + 1] + uv[ui * udims[0] + uj]);
//             expected[rj * rdims[0] + ri] = beta * (st_x + st_y);
//         }
//     REQUIRE_THAT(res.vec(), Approx(expected));
// }
