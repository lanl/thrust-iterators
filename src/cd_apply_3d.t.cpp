#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "cd_apply_3d_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 3;
using Dims = std::array<B, N>;

TEST_CASE("diffusion v1 res")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 7, j0 = 99, k0 = 12;
    const int i1 = 54, j1 = 250, k1 = 43;
    const std::array dx{0.98989, 0.15, 0.13};
    const int ugcw = pick(2, 10);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);
    const int agcw = pick(3, 9);

    vec a(B{k0 - agcw, k1 + agcw}, B{j0 - agcw, j1 + agcw}, B{i0 - agcw, i1 + agcw});
    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});
    vec ff(B{k0 - fgcw, k1 + fgcw}, B{j0 - fgcw, j1 + fgcw}, B{i0 - fgcw, i1 + fgcw});
    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(a.begin(), a.end(), f);
    std::generate(ff.begin(), ff.end(), f);
    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv1res3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
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
                          f1.data(),
                          f2.data(),
                          rgcw,
                          res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::diffusion_v1_res(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
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
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v2 res")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 98;
    const int i1 = 73, j1 = 129, k1 = 130;
    const std::array dx{0.15, 0.111, 0.1212};
    const int ugcw = pick(1, 3);
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});
    vec ff(B{k0 - fgcw, k1 + fgcw}, B{j0 - fgcw, j1 + fgcw}, B{i0 - fgcw, i1 + fgcw});
    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(ff.begin(), ff.end(), f);
    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv2res3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          alpha,
                          beta,
                          dx.data(),
                          ugcw,
                          u.data(),
                          fgcw,
                          ff.data(),
                          f0.data(),
                          f1.data(),
                          f2.data(),
                          rgcw,
                          res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::diffusion_v2_res(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
                                          alpha,
                                          beta,
                                          dx.data(),
                                          ugcw,
                                          u.data(),
                                          fgcw,
                                          ff.data(),
                                          f0.data(),
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("poisson v1 res")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 32;
    const int i1 = 73, j1 = 129, k1 = 75;
    const std::array dx{0.15, 0.987654321, 0.13};
    const int fgcw = pick(1, 4);
    const int rgcw = pick(2, 5);

    vec ff(B{k0 - fgcw, k1 + fgcw}, B{j0 - fgcw, j1 + fgcw}, B{i0 - fgcw, i1 + fgcw});
    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(ff.begin(), ff.end(), f);
    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);

    T beta = f();
    T alpha = f();

    cellpoissonv1res3d_(i0,
                        i1,
                        j0,
                        j1,
                        k0,
                        k1,
                        beta,
                        dx.data(),
                        fgcw,
                        ff.data(),
                        f0.data(),
                        f1.data(),
                        f2.data(),
                        rgcw,
                        res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::poisson_v1_res(i0,
                                        i1,
                                        j0,
                                        j1,
                                        k0,
                                        k1,
                                        beta,
                                        dx.data(),
                                        fgcw,
                                        ff.data(),
                                        f0.data(),
                                        f1.data(),
                                        f2.data(),
                                        rgcw,
                                        &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v1 apply")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 55;
    const int i1 = 73, j1 = 129, k1 = 82;
    const std::array dx{0.123456789, 0.15, 0.13};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(2, 5);
    const int agcw = pick(1, 7);

    vec a(B{k0 - agcw, k1 + agcw}, B{j0 - agcw, j1 + agcw}, B{i0 - agcw, i1 + agcw});
    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});
    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(a.begin(), a.end(), f);
    std::generate(u.begin(), u.end(), f);
    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);

    T beta = f();
    T alpha = f();

    celldiffusionv1apply3d_(i0,
                            i1,
                            j0,
                            j1,
                            k0,
                            k1,
                            alpha,
                            beta,
                            dx.data(),
                            agcw,
                            a.data(),
                            ugcw,
                            u.data(),
                            f0.data(),
                            f1.data(),
                            f2.data(),
                            rgcw,
                            res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::diffusion_v1_apply(i0,
                                            i1,
                                            j0,
                                            j1,
                                            k0,
                                            k1,
                                            alpha,
                                            beta,
                                            dx.data(),
                                            agcw,
                                            a.data(),
                                            ugcw,
                                            u.data(),
                                            f0.data(),
                                            f1.data(),
                                            f2.data(),
                                            rgcw,
                                            &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("diffusion v2 apply")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 5, j0 = 12, k0 = 22;
    const int i1 = 73, j1 = 29, k1 = 32;
    const std::array dx{0.15, 0.25, 0.13};
    const int rgcw = pick(1, 4);
    const int ugcw = pick(1, 4);

    vec u(B{k0 - ugcw, k1 + ugcw}, B{j0 - ugcw, j1 + ugcw}, B{i0 - ugcw, i1 + ugcw});
    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(u.begin(), u.end(), f);
    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);
    T beta = f();
    T alpha = f();

    celldiffusionv2apply3d_(i0,
                            i1,
                            j0,
                            j1,
                            k0,
                            k1,
                            alpha,
                            beta,
                            dx.data(),
                            ugcw,
                            u.data(),
                            f0.data(),
                            f1.data(),
                            f2.data(),
                            rgcw,
                            res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::diffusion_v2_apply(i0,
                                            i1,
                                            j0,
                                            j1,
                                            k0,
                                            k1,
                                            alpha,
                                            beta,
                                            dx.data(),
                                            ugcw,
                                            u.data(),
                                            f0.data(),
                                            f1.data(),
                                            f2.data(),
                                            rgcw,
                                            &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}

TEST_CASE("poisson v2 apply")
{
    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    const int i0 = 35, j0 = 12, k0 = 45;
    const int i1 = 47, j1 = 92, k1 = 111;
    const std::array dx{0.15, 0.7, 0.13};
    const int rgcw = 2;

    vec res(B{k0 - rgcw, k1 + rgcw}, B{j0 - rgcw, j1 + rgcw}, B{i0 - rgcw, i1 + rgcw});
    vec f0(B{k0, k1}, B{j0, j1}, B{i0, i1 + 1});
    vec f1(B{i0, i1}, B{k0, k1}, B{j0, j1 + 1});
    vec f2(B{j0, j1}, B{i0, i1}, B{k0, k1 + 1});

    std::generate(f0.begin(), f0.end(), f);
    std::generate(f1.begin(), f1.end(), f);
    std::generate(f2.begin(), f2.end(), f);
    T beta = f();

    cellpoissonv2apply3d_(i0,
                          i1,
                          j0,
                          j1,
                          k0,
                          k1,
                          beta,
                          dx.data(),
                          f0.data(),
                          f1.data(),
                          f2.data(),
                          rgcw,
                          res.data());

    std::vector<T> res_cuda(res.size());
    cd_apply_3d_cuda<T>::poisson_v2_apply(i0,
                                          i1,
                                          j0,
                                          j1,
                                          k0,
                                          k1,
                                          beta,
                                          dx.data(),
                                          f0.data(),
                                          f1.data(),
                                          f2.data(),
                                          rgcw,
                                          &res_cuda[0]);

    REQUIRE_THAT(res.vec(), Approx(res_cuda));
}
