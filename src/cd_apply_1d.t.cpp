#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes1d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_apply_1d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
using Dims = std::array<B, 1>;

TEST_CASE("diffusion v1 res")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    randomize();

    const int i0 = 7;
    const int i1 = 54;
    const std::array dx{0.15};
    const int ugcw = pick(2, 10);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);
    const int agcw = pick(3, 9);

    vec a(B{i0 - agcw, i1 + agcw});
    vec u(B{i0 - ugcw, i1 + ugcw});
    vec ff(B{i0 - fgcw, i1 + fgcw});
    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(a.begin(), a.end(), f);
    std::generate(ff.begin(), ff.end(), f);
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv1res1d_(i0,
                          i1,
                          alpha,
                          beta,
                          dx.data(),
                          agcw,
                          a.data(),
                          ugcw,
                          u.data(),
                          fgcw,
                          ff.data(),
                          f0.data(),
                          rgcw,
                          res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::diffusion_v1_res(i0,
                                          i1,
                                          alpha,
                                          beta,
                                          dx.data(),
                                          agcw,
                                          a.data(),
                                          ugcw,
                                          u.data(),
                                          fgcw,
                                          ff.data(),
                                          f0.data(),
                                          rgcw,
                                          &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v2 res")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    randomize();

    const int i0 = 5;
    const int i1 = 73;
    const std::array dx{0.15, 0.13};
    const int ugcw = pick(1, 3);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    vec u(B{i0 - ugcw, i1 + ugcw});
    vec ff(B{i0 - fgcw, i1 + fgcw});
    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(ff.begin(), ff.end(), f);
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv2res1d_(i0,
                          i1,
                          alpha,
                          beta,
                          dx.data(),
                          ugcw,
                          u.data(),
                          fgcw,
                          ff.data(),
                          f0.data(),
                          rgcw,
                          res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::diffusion_v2_res(i0,
                                          i1,
                                          alpha,
                                          beta,
                                          dx.data(),
                                          ugcw,
                                          u.data(),
                                          fgcw,
                                          ff.data(),
                                          f0.data(),
                                          rgcw,
                                          &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("poisson v1 res")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    randomize();

    const int i0 = 5;
    const int i1 = 73;
    const std::array dx{0.13};
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    vec u(B{i0 - fgcw, i1 + fgcw});
    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();
    T alpha = f();

    cellpoissonv1res1d_(
        i0, i1, beta, dx.data(), fgcw, u.data(), f0.data(), rgcw, res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::poisson_v1_res(
        i0, i1, beta, dx.data(), fgcw, u.data(), f0.data(), rgcw, &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v1 apply")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    randomize();

    const int i0 = 5;
    const int i1 = 73;
    const std::array dx{0.13};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(2, 5);
    const int agcw = pick(1, 7);

    vec a(B{i0 - agcw, i1 + agcw});
    vec u(B{i0 - ugcw, i1 + ugcw});
    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    std::generate(a.begin(), a.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv1apply1d_(i0,
                            i1,
                            alpha,
                            beta,
                            dx.data(),
                            agcw,
                            a.data(),
                            ugcw,
                            u.data(),
                            f0.data(),
                            rgcw,
                            res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::diffusion_v1_apply(i0,
                                            i1,
                                            alpha,
                                            beta,
                                            dx.data(),
                                            agcw,
                                            a.data(),
                                            ugcw,
                                            u.data(),
                                            f0.data(),
                                            rgcw,
                                            &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v2 apply")
{
    using T = double;
    using vec = md_host_vector<T, 1>;
    randomize();

    const int i0 = 5;
    const int i1 = 73;
    const std::array dx{0.11};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(1, 4);

    vec u(B{i0 - ugcw, i1 + ugcw});
    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv2apply1d_(
        i0, i1, alpha, beta, dx.data(), ugcw, u.data(), f0.data(), rgcw, res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::diffusion_v2_apply(
        i0, i1, alpha, beta, dx.data(), ugcw, u.data(), f0.data(), rgcw, &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("poisson v2 apply")
{
    using T = double;
    using vec = md_host_vector<T, 1>;

    const int i0 = 35;
    const int i1 = 47;
    const std::array dx{0.12};
    const int rgcw = 2;

    vec res(B{i0 - rgcw, i1 + rgcw});
    vec f0(B{i0, i1 + 1});

    randomize();
    std::generate(f0.begin(), f0.end(), f);

    T beta = f();

    cellpoissonv2apply1d_(i0, i1, beta, dx.data(), f0.data(), rgcw, res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_1d_cuda<T>::poisson_v2_apply(
        i0, i1, beta, dx.data(), f0.data(), rgcw, &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}
