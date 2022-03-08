#include "../cd_apply_2d_cuda.hpp"

#include "md_lazy_vector.hpp"

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v1_res(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& agcw,
                                           const T* a_,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto a = make_vec(a_, agcw, j, i);
    auto u = make_vec(u_, ugcw, j, i);
    auto f = make_vec(f_, fgcw, j, i);
    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j, i)(res = f + beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])) -
                            alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v2_res(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_vec(u_, ugcw, j, i);
    auto f = make_vec(f_, fgcw, j, i);
    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j, i)(res = f + beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])) - alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_2d_cuda<T>::poisson_v1_res(const int& i0,
                                         const int& i1,
                                         const int& j0,
                                         const int& j1,
                                         const T& beta,
                                         const T* dx,
                                         const int& fgcw,
                                         const T* f_,
                                         const T* f0_,
                                         const T* f1_,
                                         const int& rgcw,
                                         T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto f = make_vec(f_, fgcw, j, i);
    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j, i)(res = f + beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])));
    res.copy_to(res_);
}

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v1_apply(const int& i0,
                                             const int& i1,
                                             const int& j0,
                                             const int& j1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& agcw,
                                             const T* a_,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const T* f1_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto a = make_vec(a_, agcw, j, i);
    auto u = make_vec(u_, ugcw, j, i);
    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j,
                i)(res = -beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])) + alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_2d_cuda<T>::diffusion_v2_apply(const int& i0,
                                             const int& i1,
                                             const int& j0,
                                             const int& j1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const T* f1_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_vec(u_, ugcw, j, i);
    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j, i)(res = -beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])) + alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_2d_cuda<T>::poisson_v2_apply(const int& i0,
                                           const int& i1,
                                           const int& j0,
                                           const int& j1,
                                           const T& beta,
                                           const T* dx,
                                           const T* f0_,
                                           const T* f1_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto res = make_vec(res_, rgcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_domain(j, i)(res = -beta * (f0.grad_x(dx[0]) + f1.grad_y(dx[1])));
    res.copy_to(res_);
}

template struct cd_apply_2d_cuda<double>;
template struct cd_apply_2d_cuda<float>;
