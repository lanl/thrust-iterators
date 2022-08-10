#pragma once

template <typename T>
struct ls_apply_st_3d_cuda {

    static void at_point(const int& i0,
                         const int& i1,
                         const int& j0,
                         const int& j1,
                         const int& k0,
                         const int& k1,
                         const int& i,
                         const int& j,
                         const int& k,
                         const int& stencil_sz,
                         const int* offsets_,
                         const T* st_,
                         const T& a,
                         const T& b,
                         const int& fgcw,
                         const T* f_,
                         const int& ugcw,
                         const T* u_,
                         const int& rgcw,
                         const T* r_);
};
