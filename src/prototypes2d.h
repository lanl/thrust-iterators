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
void celldiffusionv1res2d_(const int&,
                           const int&,
                           const int&,
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
                           const double*,
                           const int&,
                           double*);

void celldiffusionv2res2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double&,
                           const double&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const double*,
                           const double*,
                           const int&,
                           double*);

void cellpoissonv1res2d_(const int&,
                         const int&,
                         const int&,
                         const int&,
                         const double&,
                         const double*,
                         const int&,
                         const double*,
                         const double*,
                         const double*,
                         const int&,
                         const double*);

void celldiffusionv1apply2d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double&,
                             const double&,
                             const double*,
                             const int&,
                             const double*,
                             const int&,
                             const double*,
                             const double*,
                             const double*,
                             const int&,
                             double*);

void celldiffusionv2apply2d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double&,
                             const double&,
                             const double*,
                             const int&,
                             const double*,
                             const double*,
                             const double*,
                             const int&,
                             double*);

void cellpoissonv2apply2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double&,
                           const double*,
                           const double*,
                           const double*,
                           const int&,
                           const double*);

void celldiffusionflux2d_(const int&,
                          const int&,
                          const int&,
                          const int&,
                          const double*,
                          const double*,
                          const double*,
                          const int&,
                          const double*,
                          double*,
                          double*);

void cellpoissonflux2d_(const int&,
                        const int&,
                        const int&,
                        const int&,
                        const double*,
                        const int&,
                        const double*,
                        double*,
                        double*);

void celldiffusionoffdiag2d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double*,
                             const double&,
                             const double*,
                             const double*,
                             const int&,
                             const double*);

void cellpoissonoffdiag2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double*,
                           const double&,
                           const int&,
                           const double*);

void celldiffusionv1diag2d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const double&,
                            const double*,
                            const int&,
                            const double*);

void celldiffusionv2diag2d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const double&,
                            const int&,
                            const double*);

void cellpoissondiag2d_(
    const int&, const int&, const int&, const int&, const int&, const double*);

void adjcelldiffusiondiag2d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const double*,
                             const double&,
                             const double*,
                             const double*,
                             const int&,
                             const double*);

void adjcellpoissondiag2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const int&,
                           const int&,
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

void adjcelldiffusioncfdiag2d_(const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const int&,
                               const double*,
                               const double&,
                               const double*,
                               const double*,
                               const int&,
                               const double*);

void adjcellpoissoncfdiag2d_(const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
                             const int&,
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

void adjcelldiffusionoffdiag2d_(const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
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
                                const double*,
                                const int&,
                                const double*);

void adjcellpoissonoffdiag2d_(const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
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

void adjcelldiffusioncfoffdiag2d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*);

void readjcelldiffusionoffdiag2d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*);

void adjcelldiffusioncfbdryrhs2d_(const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const int&,
                                  const double*,
                                  const int&,
                                  const double*,
                                  double*);

void cellsethomogenousbc2d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const int*,
                            const int*,
                            const int&,
                            double*);

void cellsetcorrectionbc2d_(const int&,
                            const int&,
                            const int&,
                            const int&,
                            const double*,
                            const int&,
                            const double*,
                            const double*,
                            const int&,
                            const double*,
                            const int*,
                            const int*,
                            const int&,
                            const int&,
                            const int&,
                            const int&,
                            const double&,
                            const double&);

void cellsetpoissoncorrectionbc2d_(const int&,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const double*,
                                   const int&,
                                   const double*,
                                   const int*,
                                   const int*,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const int&,
                                   const double&,
                                   const double&);

void cellsetinteriorcornerbc2d_(const int&,
                                const int&,
                                const int&,
                                const int&,
                                const int&,
                                const double*,
                                const double*,
                                const double*,
                                const double*,
                                const int*,
                                const int*,
                                const int&,
                                const int&,
                                const int&,
                                const int&);

/* level solver prototypes */
void adjcellcrsfinebdryrhs2d_(const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int&,
                              const int*,
                              const double*,
                              const int&,
                              const double*,
                              const double*);

void celladjustsystemrhspatch2d_(const int&,
                                 const int&,
                                 const int&,
                                 const int&,
                                 const int&,
                                 const int*,
                                 const int&,
                                 const double*,
                                 const int&,
                                 const double*,
                                 const int&,
                                 const double*);

void celljacobi2d_(const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*);

void cellgs2d_(const int&,
               const int&,
               const int&,
               const int&,
               const int&,
               const int*,
               const int&,
               const int&,
               const int&,
               const int&,
               const double*,
               const int&,
               const int&,
               const int&,
               const int&,
               const double*,
               const int&,
               const int&,
               const int&,
               const int&,
               const double*);

void cellrbgs2d_(const int&,
                 const int&,
                 const int&,
                 const int&,
                 const int&,
                 const int&,
                 const int*,
                 const int&,
                 const int&,
                 const int&,
                 const int&,
                 const double*,
                 const int&,
                 const int&,
                 const int&,
                 const int&,
                 const double*,
                 const int&,
                 const int&,
                 const int&,
                 const int&,
                 const double*);

void cells5rbgs2d_(const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*);

void cells9rbgs2d_(const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*,
                   const int&,
                   const int&,
                   const int&,
                   const int&,
                   const double*);

void cellblock1x1jacobi2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const int&,
                           const double*);

void cellblock2x2jacobi2d_(const int&,
                           const int&,
                           const int&,
                           const int&,
                           const double*,
                           const int&,
                           const double*,
                           const double*,
                           const int&,
                           const double*,
                           const double*);

void applystencilatpoint2d_(const int&,
                            const int&,
                            const int&,
                            const int&,
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

void applystencilonpatch2d_(const int&,
                            const int&,
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
}
