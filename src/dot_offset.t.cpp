#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "dot_offset_test.hpp"
#include <vector>

#include "md_device_vector.hpp"
#include "random.hpp"

#include <algorithm>

using Catch::Matchers::Approx;
using T = double;
using vec = std::vector<T>;

TEST_CASE("init 1d")
{
    randomize();

    const int i0 = 1, i1 = 10;
    const int sgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const int vgcw = pick(1, 3);
    const int noff = 3;

    const auto i = Ib{i0, i1};
    const auto w = Wb{0, noff - 1};

    auto st = make_md_vec<T>(sgcw, i, w);
    auto u = make_md_vec<T>(ugcw, i);
    auto v = make_md_vec<T>(vgcw, i);
    auto o = make_offset_vec(noff, 1);

    // three point stencil -1, 0, 1
    for (int i = 0; i < noff; i++) o.set_offset(i, i - 1);

    st.fill_random();
    u.fill_random();

    dot_offset_test<T>::init(
        st.data(), sgcw, u.data(), ugcw, v.data(), vgcw, o.data(), i0, i1, w.size());

    vec result = v;
    vec exact(result.size());
    vec uvec = u;
    vec stvec = st;
    std::vector<int> off = o;

    for (int i = i0; i <= i1; i++) {
        int vi = v.index(i);
        exact[vi] = 0;

        for (int j = 0; j < noff; j++) {
            int io = off[o.index(j, 0)];
            exact[vi] += uvec[u.index(i + io)] * stvec[st.index(i, j)];
        }
    }
    REQUIRE_THAT(result, Approx(exact));
}

TEST_CASE("init 2d")
{
    randomize();

    const int i0 = 1, i1 = 10;
    const int j0 = 3, j1 = 12;
    const int sgcw = pick(1, 3);
    const int ugcw = pick(1, 3);
    const int vgcw = pick(1, 3);
    const int noff = 7;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto w = Wb{0, noff - 1};

    auto st = make_md_vec<T>(sgcw, j, i, w);
    auto u = make_md_vec<T>(ugcw, j, i);
    auto v = make_md_vec<T>(vgcw, j, i);
    auto o = make_offset_vec(noff, 2);

    // 7 point stencil
    o.set_offset(0, 0, 0);
    o.set_offset(1, -1, 0);
    o.set_offset(2, +1, 0);
    o.set_offset(3, 0, -1);
    o.set_offset(4, 0, +1);
    o.set_offset(5, -1, +1);
    o.set_offset(6, +1, -1);

    st.fill_random();
    u.fill_random();

    dot_offset_test<T>::init(st.data(),
                             sgcw,
                             u.data(),
                             ugcw,
                             v.data(),
                             vgcw,
                             o.data(),
                             i0,
                             i1,
                             j0,
                             j1,
                             w.size());

    vec result = v;
    vec exact(result.size());
    vec uvec = u;
    vec stvec = st;
    std::vector<int> off = o;

    for (int j = j0; j <= j1; j++)
        for (int i = i0; i <= i1; i++) {
            int vi = v.index(j, i);
            exact[vi] = 0;

            for (int n = 0; n < noff; n++) {
                int io = off[o.index(n, 0)];
                int jo = off[o.index(n, 1)];
                exact[vi] += uvec[u.index(j + jo, i + io)] * stvec[st.index(j, i, n)];
            }
        }
    REQUIRE_THAT(result, Approx(exact));
}

TEST_CASE("init 3d")
{
    randomize();

    const int i0 = 1, i1 = 20;
    const int j0 = 3, j1 = 31;
    const int k0 = 45, k1 = 81;
    const int sgcw = pick(1, 4);
    const int ugcw = pick(1, 4);
    const int vgcw = pick(1, 4);
    const int noff = 27;

    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};
    const auto k = Kb{k0, k1};
    const auto w = Wb{0, noff - 1};

    auto st = make_md_vec<T>(sgcw, k, j, i, w);
    auto u = make_md_vec<T>(ugcw, k, j, i);
    auto v = make_md_vec<T>(vgcw, k, j, i);
    auto o = make_offset_vec(noff, 3);

    // 7 point stencil
    for (int ok = 0; ok < 3; ok++)
        for (int oj = 0; oj < 3; oj++)
            for (int oi = 0; oi < 3; oi++)
                o.set_offset(9 * ok + 3 * oj + oi, oi - 1, oj - 1, ok - 1);

    st.fill_random();
    u.fill_random();

    dot_offset_test<T>::init(st.data(),
                             sgcw,
                             u.data(),
                             ugcw,
                             v.data(),
                             vgcw,
                             o.data(),
                             i0,
                             i1,
                             j0,
                             j1,
                             k0,
                             k1,
                             w.size());

    vec result = v;
    vec exact(result.size());
    vec uvec = u;
    vec stvec = st;
    std::vector<int> off = o;

    for (int k = k0; k <= k1; k++)
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++) {
                int vi = v.index(k, j, i);
                exact[vi] = 0;

                for (int n = 0; n < noff; n++) {
                    int io = off[o.index(n, 0)];
                    int jo = off[o.index(n, 1)];
                    int ko = off[o.index(n, 2)];
                    exact[vi] += uvec[u.index(k + ko, j + jo, i + io)] *
                                 stvec[st.index(k, j, i, n)];
                }
            }
    REQUIRE_THAT(result, Approx(exact));
}
