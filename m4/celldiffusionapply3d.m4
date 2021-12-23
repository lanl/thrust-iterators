C Copyright 2006, The Regents of the University 
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
c  File:        solv_poissonresiduals3d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2000 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2887 $
c  Modified:    $Date: 2006-09-22 09:35:39 -0600 (Fri, 22 Sep 2006) $
c  Description: F77 routines that compute matrix entries for 3d Poisson solver. 
c
define(NDIM,3)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim3d.i)dnl
c
c
      recursive subroutine celldiffusionv1res3d(
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  alpha,beta,
     &  dx,
     &  agcw,
     &  acoef,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL alpha,beta
      integer agcw, ugcw,fgcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  acoef(CELL3d(ifirst,ilast,agcw)),
     &  u(CELL3d(ifirst,ilast,ugcw)),
     &  f(CELL3d(ifirst,ilast,fgcw)),
     &  res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
      REAL diver,lu
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               diver = beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                       (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                       (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
               lu = - diver +alpha*acoef(i,j,k)*u(i,j,k)
               res(i,j,k) = f(i,j,k) - lu 
            enddo
         enddo
      enddo
      return
      end
c
      recursive subroutine celldiffusionv2res3d(
     &  ifirst0,ilast0,
     &  ifirst1,ilast1,
     &  ifirst2,ilast2,
     &  alpha,beta,
     &  dx,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL alpha,beta
      integer ugcw,fgcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  u(CELL3d(ifirst,ilast,ugcw)),
     &  f(CELL3d(ifirst,ilast,fgcw)),
     &  res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
      REAL diver,lu
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               diver = beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                       (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                       (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
               lu = - diver +alpha*u(i,j,k)
               res(i,j,k) = f(i,j,k)-lu 
            enddo
         enddo
      enddo

      return
      end
c
      recursive subroutine cellpoissonv1res3d(
     &  ifirst0,ilast0,
     &  ifirst1,ilast1,
     &  ifirst2,ilast2,
     &  beta,
     &  dx,
     &  fgcw,
     &  f,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL beta
      REAL dx(0:NDIM-1)
      integer fgcw, rgcw
      REAL
     &  f(CELL3d(ifirst,ilast,fgcw)),
     &  res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
      REAL lu
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               lu = -beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                     (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                     (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
               res(i,j,k) = f(i,j,k) - lu 
            enddo
         enddo
      enddo

      return
      end

      recursive subroutine celldiffusionv1apply3d(
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  alpha,beta,
     &  dx,
     &  agcw,
     &  acoef,
     &  ugcw,
     &  u,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL alpha,beta
      integer agcw, ugcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  acoef(CELL3d(ifirst,ilast,agcw)),
     &  u(CELL3d(ifirst,ilast,ugcw)),
     &  res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
      REAL diver,lu
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               diver = beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                       (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                       (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
               res(i,j,k) = - diver +alpha*acoef(i,j,k)*u(i,j,k)
            enddo
         enddo
      enddo
      return
      end
c
      recursive subroutine celldiffusionv2apply3d(
     &  ifirst0,ilast0,
     &  ifirst1,ilast1,
     &  ifirst2,ilast2,
     &  alpha,beta,
     &  dx,
     &  ugcw,
     &  u,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL alpha,beta
      integer ugcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  u(CELL3d(ifirst,ilast,ugcw)),
     &  res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
      REAL diver,lu
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               diver = beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                       (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                       (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
               res(i,j,k) = - diver +alpha*u(i,j,k)
            enddo
         enddo
      enddo

      return
      end
c
      recursive subroutine cellpoissonv2apply3d(
     &  ifirst0,ilast0,
     &  ifirst1,ilast1,
     &  ifirst2,ilast2,
     &  beta,
     &  dx,
     &  flux0,flux1,flux2,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL beta
      REAL dx(0:NDIM-1)
      integer rgcw
      REAL res(CELL3d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
c
c***********************************************************************
c
      do k = ifirst2, ilast2
        do j = ifirst1, ilast1
          do i = ifirst0, ilast0
            res(i,j,k) = -beta*((flux0(i+1,j,k) - flux0(i,j,k))/dx(0) +
     &                   (flux1(j+1,k,i) - flux1(j,k,i))/dx(1) +
     &                   (flux2(k+1,i,j) - flux2(k,i,j))/dx(2))
          enddo
        enddo
      enddo

      return
      end
