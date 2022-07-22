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
c  File:        cellsmooth3d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2890 $
c  Modified:    $Date: 2006-09-22 12:42:57 -0600 (Fri, 22 Sep 2006) $
c  Description: F77 routines that compute matrix entries for 3d cell centered diffusion solver.
c
define(NDIM,3)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim3d.i)dnl
c
c
      recursive subroutine celljacobi3d(
     &  lo0,hi0,
     &  lo1, hi1,
     &  lo2, hi2,
     &  stencilsize,
     &  offset,
     &  slo0, shi0,
     &  slo1, shi1,
     &  slo2, shi2,
     &  stencil,
     &  flo0, fhi0,
     &  flo1, fhi1,
     &  flo2, fhi2,
     &  f,
     &  ulo0, uhi0,
     &  ulo1, uhi1,
     &  ulo2, uhi2,
     &  u)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer ulo0,ulo1,ulo2
      integer uhi0,uhi1,uhi2
      integer flo0,flo1,flo2
      integer fhi0,fhi1,fhi2
      integer slo0,slo1,slo2
      integer shi0,shi1,shi2
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL3d(slo,shi,0))
      REAL u(CELL3d(ulo,uhi,0))
      REAL f(CELL3d(flo,fhi,0))
      REAL r(CELL3d(lo,hi,0))
      integer i,j,k,s, io, jo,ko
      REAL w

      w=0.8D0

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0, hi0
               r(i,j,k) = f(i,j,k)
               do s=0,stencilsize-1
                  io=offset(0,s)
                  jo=offset(1,s)
                  ko=offset(2,s)
                  r(i,j,k) = r(i,j,k)-stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
               end do
c     do the diagonal scaling here 
               r(i,j,k) = r(i,j,k)/stencil(0,i,j,k)
            end do         
         end do
      end do

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0, hi0
               u(i,j,k)=u(i,j,k)+w*r(i,j,k)
            end do
         end do
      end do

      return
      end

      recursive subroutine cellgs3d(
     &     lo0,hi0,
     &     lo1, hi1,
     &     lo2, hi2,
     &     stencilsize,
     &     offset,
     &     slo0, shi0,
     &     slo1, shi1,
     &     slo2, shi2,
     &     stencil,
     &     flo0, fhi0,
     &     flo1, fhi1,
     &     flo2, fhi2,
     &     f,
     &     ulo0, uhi0,
     &     ulo1, uhi1,
     &     ulo2, uhi2,
     &     u)
c***********************************************************************
      implicit none
c
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer ulo0,ulo1,ulo2
      integer uhi0,uhi1,uhi2
      integer flo0,flo1,flo2
      integer fhi0,fhi1,fhi2
      integer slo0,slo1,slo2
      integer shi0,shi1,shi2
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL3d(slo,shi,0))
      REAL u(CELL3d(ulo,uhi,0))
      REAL f(CELL3d(flo,fhi,0))
      integer i,j,k
      integer s
      integer io, jo, ko
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0, hi0
               r = f(i,j,k)
               do s=1,stencilsize-1
                  io=offset(0,s)
                  jo=offset(1,s)
                  ko=offset(2,s)
                  r = r-stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
               end do
               u(i,j,k)=r/stencil(0,i,j,k)
            end do         
         end do
      end do

      return
      end

      recursive subroutine cellrbgs3d(
     &     lo0,hi0,
     &     lo1, hi1,
     &     lo2, hi2,
     &     color,
     &     stencilsize,
     &     offset,
     &     slo0, shi0,
     &     slo1, shi1,
     &     slo2, shi2,
     &     stencil,
     &     flo0, fhi0,
     &     flo1, fhi1,
     &     flo2, fhi2,
     &     f,
     &     ulo0, uhi0,
     &     ulo1, uhi1,
     &     ulo2, uhi2,
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer ulo0,ulo1,ulo2
      integer uhi0,uhi1,uhi2
      integer flo0,flo1,flo2
      integer fhi0,fhi1,fhi2
      integer slo0,slo1,slo2
      integer shi0,shi1,shi2
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL3d(slo,shi,0))
      REAL u(CELL3d(ulo,uhi,0))
      REAL f(CELL3d(flo,fhi,0))
      integer i,j,k
      integer s
      integer io, jo, ko
      REAL r

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0+mod(abs(j+k)+color,2), hi0,2
               r = f(i,j,k)
               do s=1,stencilsize-1
                  io=offset(0,s)
                  jo=offset(1,s)
                  ko=offset(2,s)
                  r = r-stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
               end do
               u(i,j,k)=r/stencil(0,i,j,k)
            end do         
         end do
      end do

      return
      end

      recursive subroutine cells7rbgs3d(
     &     lo0,hi0,
     &     lo1, hi1,
     &     lo2, hi2,
     &     color,
     &     slo0, shi0,
     &     slo1, shi1,
     &     slo2, shi2,
     &     stencil,
     &     flo0, fhi0,
     &     flo1, fhi1,
     &     flo2, fhi2,
     &     f,
     &     ulo0, uhi0,
     &     ulo1, uhi1,
     &     ulo2, uhi2,
     &     u)
c***********************************************************************
      implicit none
c
      integer color
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer ulo0,ulo1,ulo2
      integer uhi0,uhi1,uhi2
      integer flo0,flo1,flo2
      integer fhi0,fhi1,fhi2
      integer slo0,slo1,slo2
      integer shi0,shi1,shi2
      REAL stencil(0:6,CELL3d(slo,shi,0))
      REAL u(CELL3d(ulo,uhi,0))
      REAL f(CELL3d(flo,fhi,0))
      integer i,j,k

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0+mod(abs(j+k)+color,2), hi0,2
               u(i,j,k) =  f(i,j,k)-stencil(1,i,j,k)*u(i-1,j,k)
     &                             -stencil(2,i,j,k)*u(i+1,j,k)
     &                             -stencil(3,i,j,k)*u(i,j-1,k)
     &                             -stencil(4,i,j,k)*u(i,j+1,k)
     &                             -stencil(5,i,j,k)*u(i,j,k-1)
     &                             -stencil(6,i,j,k)*u(i,j,k+1)
               u(i,j,k) = u(i,j,k)/stencil(0,i,j,k)
            end do         
         end do
      end do

      return
      end

      recursive subroutine ccellblkjacobi3d(
     &     lo0,hi0,
     &     lo1, hi1,
     &     lo2, hi2,
     &     size,
     &     offset,
     &     stencil,
     &     ndof,
     &     fgcw,
     &     f,
     &     ugcw,
     &     u,
     &     rgcw,
     &     r     )
c***********************************************************************
      implicit none
c
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer size
      integer ndof
      integer offset(NDIM,size)
      integer fgcw
      REAL f(ndof,CELL3d(lo,hi,fgcw))
      integer ugcw
      REAL u(ndof,CELL3d(lo,hi,ugcw))
      integer rgcw
      REAL r(ndof,CELL3d(lo,hi,rgcw))
      REAL stencil(size, ndof*ndof,CELL3d(lo,hi,0))
      REAL block(ndof,ndof)
     
      integer i,j,k, s, io, jo,ko
      integer info
      integer ipiv(ndof)

      external dgesv
c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

      do k = lo2, hi2
         do j = lo1, hi1
            do i = lo0, hi0
	       r(:,i,j,k)=f(:,i,j,k)
               do s=2,size
                  io=offset(1,s)
                  jo=offset(2,s)
                  ko=offset(3,s)
		  block(:,:) = reshape(stencil(s,:,i,j,k), (/ndof,ndof/))
		  r(:,i,j,k)=r(:,i,j,k)-matmul(block, u(:,i+io,j+jo,k+ko))
	       end do
	       block(:,:) = reshape(stencil(1,:,i,j,k), (/ndof,ndof/))
	       call dgesv(ndof,1,block,ndof,ipiv,r(:,i,j,k),ndof,info)
	       u(:,i,j,k)=u(:,i,j,k)+r(:,i,j,k)
            end do
         end do
      end do
	
      return
      end      

      
