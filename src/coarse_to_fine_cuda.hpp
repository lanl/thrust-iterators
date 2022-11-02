// Copyright (c) 2022. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
// operated by Triad National Security, LLC for the U.S. Department of
// Energy/National Nuclear Security Administration. All rights in the
// program are reserved by Triad National Security, LLC, and the
// U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others acting
// on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute
// copies to the public, perform publicly and display publicly, and to
// permit others to do so.


#pragma once

template <typename T>
struct coarse_to_fine {
    static void copy(const int& ci0,
                     const int& ci1,
                     const int& cj0,
                     const int& cj1,
                     const int& ck0,
                     const int& ck1,
                     const int& fi0,
                     const int& fi1,
                     const int& fj0,
                     const int& fj1,
                     const int& fk0,
                     const int& fk1,
                     const int& axis,
                     const int& cbi0,
                     const int& cbi1,
                     const int& cbj0,
                     const int& cbj1,
                     const int& cbk0,
                     const int& cbk1,
                     const int& fbi0,
                     const int& fbi1,
                     const int& fbj0,
                     const int& fbj1,
                     const int& fbk0,
                     const int& fbk1,
                     const int& gcw,
                     const int* ratio,
                     const T* cdata,
                     T* fdata);

    static void copy_corner(const int& ci0,
                            const int& ci1,
                            const int& cj0,
                            const int& cj1,
                            const int& ck0,
                            const int& ck1,
                            const int& fi0,
                            const int& fi1,
                            const int& fj0,
                            const int& fj1,
                            const int& fk0,
                            const int& fk1,
                            const int& cbi0,
                            const int& cbi1,
                            const int& cbj0,
                            const int& cbj1,
                            const int& cbk0,
                            const int& cbk1,
                            const int& fbi0,
                            const int& fbi1,
                            const int& fbj0,
                            const int& fbj1,
                            const int& fbk0,
                            const int& fbk1,
                            const int& gcw,
                            const T* cdata,
                            T* fdata);
};
