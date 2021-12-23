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
      recursive subroutine nodediffusionflux3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  dx,
     &  b0,b1,b2,
     &  gcw,
     &  u,
     &  flux0,flux1,flux2)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2
      REAL dx(0:NDIM-1)
      integer gcw
      REAL
     &  u(CELL3d(ifirst,ilast,gcw))
      REAL
     &  b0(FACE3d0(ifirst,ilast,0)),
     &  b1(FACE3d1(ifirst,ilast,0)),
     &  b2(FACE3d2(ifirst,ilast,0)),
     &  flux0(FACE3d0(ifirst,ilast,0)),
     &  flux1(FACE3d1(ifirst,ilast,0)),
     &  flux2(FACE3d2(ifirst,ilast,0))
      integer i,j,k
c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0+1
               flux0(i,j,k) = b0(i,j,k)*(u(i,j,k)-u(i-1,j,k))/dx(0)
            enddo
         enddo
      enddo

      do i = ifirst0, ilast0
         do k = ifirst2, ilast2
            do j = ifirst1, ilast1+1
               flux1(j,k,i) = b1(j,k,i)*(u(i,j,k)-u(i,j-1,k))/dx(1)
            enddo
         enddo
      enddo

      do j = ifirst1, ilast1
         do i = ifirst0, ilast0
            do k = ifirst2, ilast2+1
               flux2(k,i,j) = b1(k,i,j)*(u(i,j,k)-u(i,j,k-1))/dx(2)
            enddo
         enddo
      enddo

      return
      end
c
      recursive subroutine nodepoissonflux3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  dx,
     &  gcw,
     &  u,
     &  flux0,flux1,flux2)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      REAL dx(0:NDIM-1)
      integer gcw
      REAL u(CELL3d(ifirst,ilast,gcw))
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
            do i = ifirst0, ilast0+1
               flux0(i,j,k) = (u(i,j,k)-u(i-1,j,k))/dx(0)
            enddo
         enddo
      enddo

      do i = ifirst0, ilast0
         do k = ifirst2, ilast2
            do j = ifirst1, ilast1+1
               flux1(j,k,i) = (u(i,j,k)-u(i,j-1,k))/dx(1)
            enddo
         enddo
      enddo

      do j = ifirst1, ilast1
         do i = ifirst0, ilast0
            do k = ifirst2, ilast2+1
               flux2(k,i,j) = (u(i,j,k)-u(i,j,k-1))/dx(2)
            enddo
         enddo
      enddo

      return
      end
