#include "../cd_flux_1d_cuda.hpp"

#include "md_lazy_vector.hpp"

template <typename T>
void cdf_1d_cuda<T>::flux(const int& i0,
                          const int& i1,
                          const T* dx,
                          const T* b0_,
                          const int& gcw,
                          const T* u_,
                          T* f0_)
{
    const auto i = Ib{i0, i1};

    auto u = make_vec(u_, gcw, i);
    auto f0 = make_vec(f0_, i + 1);
    auto b0 = make_vec(b0_, i + 1);

    with_lhs_domain(f0 = b0 * u.grad_x(dx[0], down));

    f0.copy_to(f0_);
}

template <typename T>
void cdf_1d_cuda<T>::poisson_flux(
    const int& i0, const int& i1, const T* dx, const int& gcw, const T* u_, T* f0_)
{
    const auto i = Ib{i0, i1};

    auto u = make_vec(u_, gcw, i);
    auto f0 = make_vec(f0_, i + 1);

    with_lhs_domain(f0 = u.grad_x(dx[0], down));

    f0.copy_to(f0_);
}

template struct cdf_1d_cuda<double>;
template struct cdf_1d_cuda<float>;
