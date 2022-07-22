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
c  File:        cellsmooth2d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2784 $
c  Modified:    $Date: 2006-08-01 15:22:35 -0600 (Tue, 01 Aug 2006) $
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
      recursive subroutine celljacobi2d(
     &  lo0,hi0,lo1, hi1,
     &  stencilsize,
     &  offset,
     &  slo0,shi0,slo1, shi1,
     &  stencil,
     &  flo0,fhi0,flo1, fhi1,
     &  f,
     &  ulo0,uhi0,ulo1, uhi1,
     &  u)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,hi0,hi1
      integer ulo0,ulo1,uhi0,uhi1
      integer flo0,flo1,fhi0,fhi1
      integer slo0,slo1,shi0,shi1
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL2d(slo,shi,0))
      REAL u(CELL2d(ulo,uhi,0))
      REAL f(CELL2d(flo,fhi,0))
      REAL r(CELL2d(lo,hi,0))
      integer i,j,s, io, jo
      REAL w

      w=0.8

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0, hi0
            r(i,j) = f(i,j)
            do s=0,stencilsize-1
               io=offset(0,s)
               jo=offset(1,s)
               r(i,j) = r(i,j)-stencil(s,i,j)*u(i+io,j+jo)
            end do
c do the diagonal scaling here 
            r(i,j) = r(i,j)/stencil(0,i,j)
         end do         
      end do

      do j = lo1, hi1
         do i = lo0, hi0
            u(i,j)=u(i,j)+w*r(i,j)
         end do
      end do

      return
      end

      recursive subroutine cellgs2d(
     &     lo0,hi0,lo1, hi1,
     &     stencilsize,
     &     offset,
     &     slo0,shi0,slo1, shi1,
     &     stencil,
     &     flo0,fhi0,flo1, fhi1,
     &     f,
     &     ulo0,uhi0,ulo1, uhi1,
     &     u)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,hi0,hi1
      integer ulo0,ulo1,uhi0,uhi1
      integer flo0,flo1,fhi0,fhi1
      integer slo0,slo1,shi0,shi1
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL2d(slo,shi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL2d(ulo,uhi,0))
      REAL f(CELL2d(flo,fhi,0))
      integer i,j,s, io, jo
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0, hi0
            r = f(i,j)
            do s=1,stencilsize-1
               io=offset(0,s)
               jo=offset(1,s)
               r = r-stencil(s,i,j)*u(i+io,j+jo)
            end do
            u(i,j)=r/stencil(0,i,j)
         end do         
      end do

      return
      end

      recursive subroutine cellrbgs2d(
     &     lo0,hi0,lo1, hi1,
     &     color,
     &     stencilsize,
     &     offset,
     &     slo0,shi0,slo1, shi1,
     &     stencil,
     &     flo0,fhi0,flo1, fhi1,
     &     f,
     &     ulo0,uhi0,ulo1, uhi1,
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,lo1,hi0,hi1
      integer ulo0,ulo1,uhi0,uhi1
      integer flo0,flo1,fhi0,fhi1
      integer slo0,slo1,shi0,shi1
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL2d(slo,shi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL2d(ulo,uhi,0))
      REAL f(CELL2d(flo,fhi,0))
      integer i,j,s, io, jo
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0+mod(j+color,2), hi0,2
            r = f(i,j)
            do s=1,stencilsize-1
               io=offset(0,s)
               jo=offset(1,s)
               r = r-stencil(s,i,j)*u(i+io,j+jo)
            end do
            u(i,j)=r/stencil(0,i,j)
         end do         
      end do

      return
      end

      recursive subroutine cells5rbgs2d(
     &     lo0,hi0,lo1, hi1,
     &     color,
     &     slo0,shi0,slo1, shi1,
     &     stencil,
     &     flo0,fhi0,flo1, fhi1,
     &     f,
     &     ulo0,uhi0,ulo1, uhi1,
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,lo1,hi0,hi1
      integer ulo0,ulo1,uhi0,uhi1
      integer flo0,flo1,fhi0,fhi1
      integer slo0,slo1,shi0,shi1
      REAL stencil(0:4,CELL2d(slo,shi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL2d(ulo,uhi,0))
      REAL f(CELL2d(flo,fhi,0))
      integer i,j

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0+mod(j+color,2), hi0,2
            u(i,j) = (f(i,j)-stencil(1,i,j)*u(i-1,j)
     &                -stencil(2,i,j)*u(i+1,j)
     &                -stencil(3,i,j)*u(i,j-1)
     &                -stencil(4,i,j)*u(i,j+1))/stencil(0,i,j)
         end do         
      end do

      return
      end

      recursive subroutine cells9rbgs2d(
     &     lo0,hi0,lo1, hi1,
     &     color,
     &     slo0,shi0,slo1, shi1,
     &     stencil,
     &     flo0,fhi0,flo1, fhi1,
     &     f,
     &     ulo0,uhi0,ulo1, uhi1,
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,lo1,hi0,hi1
      integer ulo0,ulo1,uhi0,uhi1
      integer flo0,flo1,fhi0,fhi1
      integer slo0,slo1,shi0,shi1
      REAL stencil(0:8,CELL2d(slo,shi,0))
      integer fgcw
      integer ugcw
      REAL u(CELL2d(ulo,uhi,0))
      REAL f(CELL2d(flo,fhi,0))
      integer i,j

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do j = lo1, hi1
         do i = lo0+mod(j+color,2), hi0,2
            u(i,j) = (f(i,j)-stencil(1,i,j)*u(i-1,j+0)
     &                      -stencil(2,i,j)*u(i+1,j+0)
     &                      -stencil(3,i,j)*u(i+0,j-1)
     &                      -stencil(4,i,j)*u(i+0,j+1)
     &                      -stencil(5,i,j)*u(i-1,j-1)
     &                      -stencil(6,i,j)*u(i+1,j-1)
     &                      -stencil(7,i,j)*u(i-1,j+1)
     &                      -stencil(8,i,j)*u(i+1,j+1))/stencil(0,i,j)
         end do         
      end do

      return
      end
