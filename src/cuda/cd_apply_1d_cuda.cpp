#include "../cd_apply_1d_cuda.hpp"

#include "md_lazy_vector.hpp"

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v1_res(const int& i0,
                                           const int& i1,
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
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto a = make_vec(a_, agcw, i);
    auto u = make_vec(u_, ugcw, i);
    auto f = make_vec(f_, fgcw, i);
    auto res = make_vec(res_, rgcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])) - alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v2_res(const int& i0,
                                           const int& i1,
                                           const T& alpha,
                                           const T& beta,
                                           const T* dx,
                                           const int& ugcw,
                                           const T* u_,
                                           const int& fgcw,
                                           const T* f_,
                                           const T* f0_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_vec(u_, ugcw, i);
    auto f = make_vec(f_, fgcw, i);
    auto res = make_vec(res_, rgcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])) - alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::poisson_v1_res(const int& i0,
                                         const int& i1,
                                         const T& beta,
                                         const T* dx,
                                         const int& fgcw,
                                         const T* f_,
                                         const T* f0_,
                                         const int& rgcw,
                                         T* res_)
{
    const auto i = Ib{i0, i1};

    auto f = make_vec(f_, fgcw, i);
    auto res = make_vec(res_, rgcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = f + beta * (f0.grad_x(dx[0])));
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v1_apply(const int& i0,
                                             const int& i1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& agcw,
                                             const T* a_,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};

    auto a = make_vec(a_, agcw, i);
    auto u = make_vec(u_, ugcw, i);
    auto res = make_vec(res_, rgcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])) + alpha * a * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::diffusion_v2_apply(const int& i0,
                                             const int& i1,
                                             const T& alpha,
                                             const T& beta,
                                             const T* dx,
                                             const int& ugcw,
                                             const T* u_,
                                             const T* f0_,
                                             const int& rgcw,
                                             T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_vec(u_, ugcw, i);
    auto res = make_vec(res_, rgcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])) + alpha * u);
    res.copy_to(res_);
}

template <typename T>
void cd_apply_1d_cuda<T>::poisson_v2_apply(const int& i0,
                                           const int& i1,
                                           const T& beta,
                                           const T* dx,
                                           const T* f0_,
                                           const int& rgcw,
                                           T* res_)
{
    const auto i = Ib{i0, i1};

    auto res = make_vec(res_, rgcw,  i);
    auto f0 = make_vec(f0_, i + 1);

    with_domain(i)(res = -beta * (f0.grad_x(dx[0])));
    res.copy_to(res_);
}

template struct cd_apply_1d_cuda<double>;
template struct cd_apply_1d_cuda<float>;
