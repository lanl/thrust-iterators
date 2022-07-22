C Copyright 2005, The Regents of the University 
C of California. This software was produced under
C a U.S. Government contract (W-7405-ENG-36) 
C by Los Alamos National Laboratory, which is
C operated by the University of California for the
C U.S. Department of Energy. The U.S.
C Government is licensed to use, reproduce, and
C distribute this software. Permission is granted
C to the public to copy and use this software
C without charge, provided that this Notice and
C any statement of authorship are reproduced on
C all copies. Neither the Government nor the
C University makes any warranty, express or
C implied, or assumes any liability or
C responsibility for the use of this software.
C

c
c  File:        celldiffusionstencilcoeffs1d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2533 $
c  Modified:    $Date: 2006-05-11 15:57:25 -0600 (Thu, 11 May 2006) $
c  Description: F77 routines that compute matrix entries for 1d cell centered diffusion solver.
c
define(NDIM,1)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim1d.i)dnl
c
c
      recursive subroutine cellblock1x1jacobi1d(
     &  lo0,hi0,
     &  stencil,
     &  fgcw,
     &  f,
     &  ugcw, 
     &  u)
c***********************************************************************
      implicit none
c
      integer lo0,hi0
      REAL stencil(CELL1d(lo,hi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL1d(lo,hi,ugcw))
      REAL f(CELL1d(lo,hi,fgcw))
      integer i
      REAL w

      w=2.0d0/3.0d0

c we make the assumption that the diagonal entry for the stencil is
c always stored last in the stencil

      do i = lo0, hi0
         u(i)=u(i)+w*f(i)/stencil(i)
      end do         

      return
      end

      recursive subroutine cellblock2x2jacobi1d(
     &  lo0,hi0,
     &  a,
     &  fgcw,
     &  f0,f1,
     &  ugcw, 
     &  u0,u1)
c***********************************************************************
      implicit none
c
      integer lo0,hi0
      REAL a(0:3,CELL1d(lo,hi,0))
      integer fgcw
      integer ugcw
      REAL u0(CELL1d(lo,hi,ugcw))
      REAL u1(CELL1d(lo,hi,ugcw))
      REAL f0(CELL1d(lo,hi,fgcw))
      REAL f1(CELL1d(lo,hi,fgcw))
      integer i
      REAL det
      REAL w

      w=2.0d0/3.0d0

c we make the assumption that the diagonal entry for the stencil is
c always stored last in the stencil

      do i = lo0, hi0
         det=a(0,i)*a(3,i)-a(1,i)*a(2,i)
         u0(i)=u0(i)+w*(a(3,i)*f0(i)-a(1,i)*f1(i))/det
         u1(i)=u1(i)+w*(a(0,i)*f1(i)-a(2,i)*f0(i))/det
      end do

      return
      end
