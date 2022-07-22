/*
Copyright 2005, The Regents of the University
of California. This software was produced under
a U.S. Government contract (W-7405-ENG-36)
by Los Alamos National Laboratory, which is
operated by the University of California for the
U.S. Department of Energy. The U.S.
Government is licensed to use, reproduce, and
distribute this software. Permission is granted
to the public to copy and use this software
without charge, provided that this Notice and
any statement of authorship are reproduced on
all copies. Neither the Government nor the
University makes any warranty, express or
implied, or assumes any liability or
responsibility for the use of this software.
*/

extern "C" {
void celldiffusionoffdiag1d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double*,
                             const double&,
                             const double*,
                             const int&,
                             const double*);

void cellpoissonoffdiag1d_(
    const int&, const int&, const double*, const double&, const int&, const double*);

void celldiffusionv1diag1d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const double&,
                            const double*,
                            const int&,
                            const double*);

void celldiffusionv2diag1d_(
    const int&, const int&, const double&, const int&, const double*);

void cellpoissondiag1d_(const int&, const int&, const int&, const double*);

void adjcelldiffusiondiag1d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double*,
                             const double&,
                             const double*,
                             const int&,
                             const double*);

void adjcellpoissondiag1d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double*,
                           const double&,
                           const int&,
                           const double*);

void adjcelldiffusioncfdiag1d_(const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const double*,
                               const double&,
                               const double*,
                               const int&,
                               const double*);

void adjcellpoissoncfdiag1d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double*,
                             const double&,
                             const int&,
                             const double*);

void adjcelldiffusionoffdiag1d_(const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
                                const double*,
                                const double&,
                                const double&,
                                const double&,
                                const double*,
                                const int&,
                                const double*);

void adjcellpoissonoffdiag1d_(const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const double*,
                              const double&,
                              const double&,
                              const double&,
                              const int&,
                              const double*);

void adjcelldiffusioncfoffdiag1d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*);

void readjcelldiffusionoffdiag1d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*);

void adjcelldiffusioncfbdryrhs1d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*,
                                  const int&,
                                  const double*,
                                  double*);

void celldiffusionflux1d_(const int&,
                          const int&,
                          const double*,
                          const double*,
                          const int&,
                          const double*,
                          double*);

void cellpoissonflux1d_(
    const int&, const int&, const double*, const int&, const double*, double*);

void celldiffusionv1res1d_(const int&,
                           const int&,
                           const double&,
                           const double&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const double*,
                           const int&,
                           double*);

void celldiffusionv2res1d_(const int&,
                           const int&,
                           const double&,
                           const double&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const double*,
                           const int&,
                           double*);

void cellpoissonv1res1d_(const int&,
                         const int&,
                         const double&,
                         const double*,
                         const int&,
                         const double*,
                         const double*,
                         const int&,
                         const double*);

void celldiffusionv1apply1d_(const int&,
                             const int&,
                             const double&,
                             const double&,
                             const double*,
                             const int&,
                             const double*,
                             const int&,
                             const double*,
                             const double*,
                             const int&,
                             double*);

void celldiffusionv2apply1d_(const int&,
                             const int&,
                             const double&,
                             const double&,
                             const double*,
                             const int&,
                             const double*,
                             const double*,
                             const int&,
                             double*);

void cellpoissonv2apply1d_(const int&,
                           const int&,
                           const double&,
                           const double*,
                           const double*,
                           const int&,
                           const double*);

void cellsetcorrectionbc1d_(const int&,
                            const int&,
                            const double*,
                            const int&,
                            const double*,
                            const int&,
                            const double*,
                            const int*,
                            const int*,
                            const int&,
                            const int&,
                            const int&,
                            const double&,
                            const double&);

void cellsetpoissoncorrectionbc1d_(const int&,
                                   const int&,
                                   const double*,
                                   const int&,
                                   const double*,
                                   const int*,
                                   const int*,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const double&,
                                   const double&);

/* solvers/level prototypes */
void adjcellcrsfinebdryrhs1d_(const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int*,
                              const double*,
                              const int&,
                              const double*,
                              double*);

void celladjustsystemrhspatch1d_(const int&,
                                 const int&,
                                 const int&,
                                 const int*,
                                 const int&,
                                 const double*,
                                 const int&,
                                 const double*,
                                 const int&,
                                 const double*);

void celljacobi1d_(const int&,
                   const int&,
                   const int&,
                   const int*,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const double*);

void cellgs1d_(const int&,
               const int&,
               const int&,
               const int*,
               const int&,
               const int&,
               const double*,
               const int&,
               const int&,
               const double*,
               const int&,
               const int&,
               const double*);

void cellrbgs1d_(const int&,
                 const int&,
                 const int&,
                 const int&,
                 const int*,
                 const int&,
                 const int&,
                 const double*,
                 const int&,
                 const int&,
                 const double*,
                 const int&,
                 const int&,
                 const double*);

void cellblock1x1jacobi1d_(const int&,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*);

void cellblock2x2jacobi1d_(const int&,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const double*,
                           const int&,
                           const double*,
                           const double*);

void applystencilatpoint1d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int*,
                            const double*,
                            const double&,
                            const double&,
                            const int&,
                            const double*,
                            const int&,
                            const double*,
                            const int&,
                            const double*);

void applystencilonpatch1d_(const int&,
                            const int&,
                            const int&,
                            const int*,
                            const double*,
                            const double&,
                            const double&,
                            const int&,
                            const double*,
                            const int&,
                            const double*,
                            const int&,
                            const double*);
}
