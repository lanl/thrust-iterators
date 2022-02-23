#include "../lazy_vec.hpp"

#include "md_lazy_vector.hpp"

using namespace lazy::placeholders;

template <typename T>
void lazy_vec_cuda<T>::init(const int& i0,
                            const int& i1,
                            const T& beta,
                            const T* dx,
                            const int ugcw,
                            const T* u_,
                            const int& rgcw,
                            T* res_)
{
    const auto i = Ib{i0, i1};

    auto u = make_vec(u_, ugcw, i);
    auto res = make_vec(res_, rgcw, i);

    with_domain(I = i)(res = 2 * (u.grad_x(dx[0], down) + u.grad_x(dx[0])) / (u + 10));

    res.copy_to(res_);
}

template <typename T>
void lazy_vec_cuda<T>::init(const int& i0,
                            const int& i1,
                            const int& j0,
                            const int& j1,
                            const T& beta,
                            const T* dx,
                            const int ugcw,
                            const T* u_,
                            const int& rgcw,
                            T* res_)
{
    const auto i = Ib{i0, i1};
    const auto j = Jb{j0, j1};

    auto u = make_vec(u_, ugcw, i, j);
    auto res = make_vec(res_, rgcw, j, i);

    with_domain(J = j, I = i)(res = 3 * (u.grad_y(dx[1]) + u.grad_x(dx[0])));

    res.copy_to(res_);
}

template struct lazy_vec_cuda<double>;
