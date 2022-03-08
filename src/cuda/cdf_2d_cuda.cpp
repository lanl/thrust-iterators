#include "../cd_flux_2d_cuda.hpp"

#include "md_lazy_vector.hpp"

template <typename T>
void cdf_2d_cuda<T>::flux(const int& i0,
                          const int& j0,
                          const int& i1,
                          const int& j1,
                          const T* dx,
                          const T* b0_,
                          const T* b1_,
                          const int& gcw,
                          const T* u_,
                          T* f0_,
                          T* f1_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_vec(u_, gcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto b0 = make_vec(b0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);
    auto b1 = make_vec(b1_, i, j + 1);

    with_lhs_domain(f0 = b0 * u.grad_x(dx[0], down));
    with_lhs_domain(f1 = b1 * u.grad_y(dx[1], down));

    f0.copy_to(f0_);
    f1.copy_to(f1_);
}

template <typename T>
void cdf_2d_cuda<T>::poisson_flux(const int& i0,
                                  const int& j0,
                                  const int& i1,
                                  const int& j1,
                                  const T* dx,
                                  const int& gcw,
                                  const T* u_,
                                  T* f0_,
                                  T* f1_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_vec(u_, gcw, j, i);
    auto f0 = make_vec(f0_, j, i + 1);
    auto f1 = make_vec(f1_, i, j + 1);

    with_lhs_domain(f0 = u.grad_x(dx[0], down));
    with_lhs_domain(f1 = u.grad_y(dx[1], down));

    f0.copy_to(f0_);
    f1.copy_to(f1_);
}

template struct cdf_2d_cuda<double>;
template struct cdf_2d_cuda<float>;
