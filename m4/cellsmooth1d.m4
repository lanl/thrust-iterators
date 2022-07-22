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
c  File:        cellsmooth1d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2727 $
c  Modified:    $Date: 2006-06-22 15:52:36 -0600 (Thu, 22 Jun 2006) $
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
      recursive subroutine celljacobi1d(
     &  lo0,hi0,
     &  stencilsize,
     &  offset,
     &  slo0, shi0,
     &  stencil,
     &  flo0, fhi0,
     &  f,
     &  ulo0, uhi0, 
     &  u)
c***********************************************************************
      implicit none
c
      integer lo0,hi0
      integer ulo0, uhi0
      integer flo0, fhi0
      integer slo0, shi0
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL1d(slo,shi,0))
      REAL u(CELL1d(ulo,uhi,0))
      REAL f(CELL1d(flo,fhi,0))
      REAL r(CELL1d(lo,hi,0))
      integer i,s, io
      REAL w

      w=2.0d0/3.0d0

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do i = lo0, hi0
         r(i) = f(i)
         do s=0,stencilsize-1
            io=offset(0,s)
            r(i) = r(i)-stencil(s,i)*u(i+io)
         end do
c     do the diagonal scaling here 
         r(i) = r(i)/stencil(0,i)
      end do

      do i = lo0, hi0
         u(i)=u(i)+w*r(i)
      end do

      return
      end

      recursive subroutine cellgs1d(
     &     lo0,hi0,
     &     stencilsize,
     &     offset,
     &     slo0, shi0,
     &     stencil,
     &     flo0, fhi0,
     &     f,
     &     ulo0, uhi0, 
     &     u)
c***********************************************************************
      implicit none
c
      integer lo0,hi0
      integer ulo0, uhi0
      integer flo0, fhi0
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      integer slo0, shi0
      REAL stencil(0:stencilsize-1,CELL1d(slo,shi,0))
      REAL u(CELL1d(ulo,uhi,0))
      REAL f(CELL1d(ulo,uhi,0))
      integer i,s, io
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do i = lo0, hi0
         r = f(i)
         do s=1,stencilsize-1
            io=offset(0,s)
            r = r-stencil(s,i)*u(i+io)
         end do
         u(i)=r/stencil(0,i)
      end do

      return
      end

      recursive subroutine cellrbgs1d(
     &     lo0,hi0,
     &     color,
     &     stencilsize,
     &     offset,
     &     slo0, shi0,
     &     stencil,
     &     flo0, fhi0,
     &     f,
     &     ulo0, uhi0, 
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,hi0
      integer ulo0, uhi0
      integer flo0, fhi0
      integer slo0, shi0
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL1d(slo,shi,0))
      REAL u(CELL1d(ulo,uhi,0))
      REAL f(CELL1d(flo,fhi,0))
      integer i,s, io
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do i = lo0+mod(color,2), hi0,2
         r = f(i)
         do s=1,stencilsize-1
            io=offset(0,s)
            r = r-stencil(s,i)*u(i+io)
         end do
         u(i)=r/stencil(0,i)
      end do         

      return
      end
