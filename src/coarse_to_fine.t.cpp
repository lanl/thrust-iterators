#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "prototypes3d.h"
#include "random.hpp"
#include <algorithm>

#include "coarse_to_fine_cuda.hpp"
#include "md_host_vector.hpp"

using Catch::Matchers::Approx;

constexpr auto f = []() { return pick(0.0, 1.0); };
using B = hbounds;
static constexpr auto N = 3;
using Dims = std::array<B, N>;

template <typename T>
using kernel = coarse_to_fine<T>;

TEST_CASE("copy")
{

    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    // coarse box
    const int cbi0 = 10, cbj0 = 20, cbk0 = 30;
    const int cbi1 = 30, cbj1 = 41, cbk1 = 52;
    // ratio
    std::array r{2, 3, 4};
    // fine box
    const int fbi0 = r[0] * cbi0, fbj0 = r[1] * cbj0, fbk0 = r[2] * cbk0;
    const int fbi1 = r[0] * cbi1, fbj1 = r[1] * cbj1, fbk1 = r[2] * cbk1;

    // coarse common edge
    const int cs = pick(1, 3);
    const int ci0 = cbi0 + cs, cj0 = cbj0 + cs, ck0 = cbk0 + cs;
    const int ci1 = cbi1 - cs, cj1 = cbj1 - cs, ck1 = cbk1 - cs;
    // fine common edge
    const int fi0 = r[0] * ci0, fj0 = r[1] * cj0, fk0 = r[2] * ck0;
    const int fi1 = r[0] * ci1, fj1 = r[1] * cj1, fk1 = r[2] * ck1;

    const int gcw = pick(2, 3);

    auto axis = GENERATE(0, 1, 2);

    vec cd(
        B{cbk0 - gcw, cbk1 + gcw}, B{cbj0 - gcw, cbj1 + gcw}, B{cbi0 - gcw, cbi1 + gcw});
    vec fd(
        B{fbk0 - gcw, fbk1 + gcw}, B{fbj0 - gcw, fbj1 + gcw}, B{fbi0 - gcw, fbi1 + gcw});

    std::generate(cd.begin(), cd.end(), f);
    std::generate(fd.begin(), fd.end(), f);

    std::vector<T> fd_cuda{fd.vec()};

    copycoarsetofine3d_(ci0,
                        ci1,
                        cj0,
                        cj1,
                        ck0,
                        ck1,
                        fi0,
                        fi1,
                        fj0,
                        fj1,
                        fk0,
                        fk1,
                        axis,
                        cbi0,
                        cbi1,
                        cbj0,
                        cbj1,
                        cbk0,
                        cbk1,
                        fbi0,
                        fbi1,
                        fbj0,
                        fbj1,
                        fbk0,
                        fbk1,
                        gcw,
                        r.data(),
                        cd.data(),
                        fd.data());

    kernel<T>::copy(ci0,
                    ci1,
                    cj0,
                    cj1,
                    ck0,
                    ck1,
                    fi0,
                    fi1,
                    fj0,
                    fj1,
                    fk0,
                    fk1,
                    axis,
                    cbi0,
                    cbi1,
                    cbj0,
                    cbj1,
                    cbk0,
                    cbk1,
                    fbi0,
                    fbi1,
                    fbj0,
                    fbj1,
                    fbk0,
                    fbk1,
                    gcw,
                    r.data(),
                    cd.data(),
                    &fd_cuda[0]);

    REQUIRE_THAT(fd.vec(), Approx(fd_cuda));
}

TEST_CASE("corner")
{

    using T = double;
    using vec = md_host_vector<T, N>;
    randomize();

    // coarse box
    const int cbi0 = 10, cbj0 = 20, cbk0 = 30;
    const int cbi1 = 30, cbj1 = 41, cbk1 = 52;
    // ratio
    auto r = GENERATE(2, 3, 4);
    // fine box
    const int fbi0 = r * cbi0, fbj0 = r * cbj0, fbk0 = r * cbk0;
    const int fbi1 = r * cbi1, fbj1 = r * cbj1, fbk1 = r * cbk1;

    // coarse common edge
    const int cs = pick(1, 3);
    const int ci0 = cbi0 + cs, cj0 = cbj0 + cs, ck0 = cbk0 + cs;
    const int ci1 = cbi1 - cs, cj1 = cbj1 - cs, ck1 = cbk1 - cs;
    // fine common edge
    const int fi0 = r * ci0, fj0 = r * cj0, fk0 = r * ck0;
    const int fi1 = r * ci1, fj1 = r * cj1, fk1 = r * ck1;

    const int gcw = pick(2, 3);

    vec cd(
        B{cbk0 - gcw, cbk1 + gcw}, B{cbj0 - gcw, cbj1 + gcw}, B{cbi0 - gcw, cbi1 + gcw});
    vec fd(
        B{fbk0 - gcw, fbk1 + gcw}, B{fbj0 - gcw, fbj1 + gcw}, B{fbi0 - gcw, fbi1 + gcw});

    std::generate(cd.begin(), cd.end(), f);
    std::generate(fd.begin(), fd.end(), f);

    std::vector<T> fd_cuda{fd.vec()};

    copycoarsetofinecorner3d_(ci0,
                              ci1,
                              cj0,
                              cj1,
                              ck0,
                              ck1,
                              fi0,
                              fi1,
                              fj0,
                              fj1,
                              fk0,
                              fk1,
                              cbi0,
                              cbi1,
                              cbj0,
                              cbj1,
                              cbk0,
                              cbk1,
                              fbi0,
                              fbi1,
                              fbj0,
                              fbj1,
                              fbk0,
                              fbk1,
                              gcw,
                              cd.data(),
                              fd.data());

    kernel<T>::copy_corner(ci0,
                           ci1,
                           cj0,
                           cj1,
                           ck0,
                           ck1,
                           fi0,
                           fi1,
                           fj0,
                           fj1,
                           fk0,
                           fk1,
                           cbi0,
                           cbi1,
                           cbj0,
                           cbj1,
                           cbk0,
                           cbk1,
                           fbi0,
                           fbi1,
                           fbj0,
                           fbj1,
                           fbk0,
                           fbk1,
                           gcw,
                           cd.data(),
                           &fd_cuda[0]);

    REQUIRE_THAT(fd.vec(), Approx(fd_cuda));
}
