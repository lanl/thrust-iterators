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
c  File:        solv_poissonresiduals2d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2000 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 1984 $
c  Modified:    $Date: 2005-11-02 09:54:59 -0700 (Wed, 02 Nov 2005) $
c  Description: F77 routines that compute matrix entries for 2d Poisson solver. 
c
define(NDIM,2)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim2d.i)dnl
c
c
      recursive subroutine celldiffusionflux2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  dx,
     &  b0,b1,
     &  gcw,
     &  u,
     &  flux0,flux1)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL dx(0:1)
      integer gcw
      REAL
     &  u(CELL2d(ifirst,ilast,gcw))
      REAL
     &  b0(FACE2d0(ifirst,ilast,0)),
     &  b1(FACE2d1(ifirst,ilast,0)),
     &  flux0(FACE2d0(ifirst,ilast,0)),
     &  flux1(FACE2d1(ifirst,ilast,0))
      integer ic0,ic1
      integer ie0,ie1
c
c***********************************************************************
c
      do ic1 = ifirst1, ilast1
         do ie0 = ifirst0, ilast0+1
            flux0(ie0,ic1) = b0(ie0,ic1)*(u(ie0,ic1)-u(ie0-1,ic1))/dx(0)
         enddo
      enddo

      do ic0 = ifirst0, ilast0
         do ie1 = ifirst1, ilast1+1
            flux1(ie1,ic0) = b1(ie1,ic0)*(u(ic0,ie1)-u(ic0,ie1-1))/dx(1)
         enddo
      enddo
      return
      end
c
      recursive subroutine cellpoissonflux2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  dx,
     &  gcw,
     &  u,
     &  flux0,flux1)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL dx(0:1)
      integer gcw
      REAL
     &  u(CELL2d(ifirst,ilast,gcw))
      REAL
     &  flux0(FACE2d0(ifirst,ilast,0)),
     &  flux1(FACE2d1(ifirst,ilast,0))
      integer ic0,ic1
      integer ie0,ie1
c
c***********************************************************************
c
      do ie0 = ifirst0, ilast0+1
         do ic1 = ifirst1, ilast1
            flux0(ie0,ic1) = (u(ie0,ic1)-u(ie0-1,ic1))/dx(0)
         enddo
      enddo
      do ie1 = ifirst1, ilast1+1
         do ic0 = ifirst0, ilast0
            flux1(ie1,ic0) = (u(ic0,ie1)-u(ic0,ie1-1))/dx(1)
         enddo
      enddo
      return
      end
