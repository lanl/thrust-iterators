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
c  File:        celldiffusionstencilcoeffs2d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2889 $
c  Modified:    $Date: 2006-09-22 12:42:21 -0600 (Fri, 22 Sep 2006) $
c  Description: F77 routines that compute matrix entries for 2d cell centered diffusion solver.
c
define(NDIM,2)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim2d.i)dnl
c
c
      recursive subroutine cellblock1x1jacobi2d(
     &  lo0,hi0,lo1, hi1,
     &  stencil,
     &  fgcw,
     &  f,
     &  ugcw, 
     &  u)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,hi0,hi1
      REAL stencil(CELL2d(lo,hi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL2d(lo,hi,ugcw))
      REAL f(CELL2d(lo,hi,fgcw))
      integer i,j
      REAL w

      w=0.8

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0, hi0
            u(i,j)=u(i,j)+w*f(i,j)/stencil(i,j)
         end do         
      end do

      return
      end

      recursive subroutine cellblock2x2jacobi2d(
     &  lo0,hi0,lo1, hi1,
     &  a,
     &  fgcw,
     &  f0,f1,
     &  ugcw, 
     &  u0,u1)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,hi0,hi1
      REAL a(0:3,CELL2d(lo,hi,0))
      integer fgcw
      integer ugcw
      REAL u0(CELL2d(lo,hi,ugcw))
      REAL u1(CELL2d(lo,hi,ugcw))
      REAL f0(CELL2d(lo,hi,fgcw))
      REAL f1(CELL2d(lo,hi,fgcw))
      integer i,j
      REAL det
      REAL w

      w=0.8

c we make the assumption that the diagonal entry for the stencil is
c always stored last in the stencil

      do j = lo1, hi1
         do i = lo0, hi0
            det=a(0,i,j)*a(3,i,j)-a(1,i,j)*a(2,i,j)
            u0(i,j)=u0(i,j)+w*(a(3,i,j)*f0(i,j)-a(1,i,j)*f1(i,j))/det
            u1(i,j)=u1(i,j)+w*(a(0,i,j)*f1(i,j)-a(2,i,j)*f0(i,j))/det
         end do
      end do

      return
      end
