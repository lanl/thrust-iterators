#pragma once

// dims order: k, j, i, w
template <typename T>
struct dot_offset_test {
    static void init(const T* stencil,
                     const int& sgcw,
                     const T* u,
                     const int& ugcw,
                     T* v,
                     const int& vgcw,
                     int* offset,
                     const int& i0,
                     const int& i1,
                     const int& w);

    static void init(const T* stencil,
                     const int& sgcw,
                     const T* u,
                     const int& ugcw,
                     T* v,
                     const int& vgcw,
                     int* offset,
                     const int& i0,
                     const int& i1,
                     const int& j0,
                     const int& j1,
                     const int& w);

    static void init(const T* stencil,
                     const int& sgcw,
                     const T* u,
                     const int& ugcw,
                     T* v,
                     const int& vgcw,
                     int* offset,
                     const int& i0,
                     const int& i1,
                     const int& j0,
                     const int& j1,
                     const int& k0,
                     const int& k1,
                     const int& w);
};
