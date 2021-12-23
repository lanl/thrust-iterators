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
c  Revision:    $Revision: 2531 $
c  Modified:    $Date: 2006-05-11 15:38:55 -0600 (Thu, 11 May 2006) $
c  Description: F77 routines that compute matrix entries for 2d Poisson solver. 
c
define(NDIM,1)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim1d.i)dnl
c
c
      recursive subroutine celldiffusionflux1d(
     &  ifirst0,ilast0,
     &  dx,
     &  b0,
     &  gcw,
     &  u,
     &  flux0)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL dx(0:NDIM-1)
      integer gcw
      REAL u(CELL1d(ifirst,ilast,gcw))
      REAL
     &  b0(FACE1d(ifirst,ilast,0)),
     &  flux0(FACE1d(ifirst,ilast,0))
      integer ie0
c
c***********************************************************************
c
      do ie0 = ifirst0, ilast0+1
         flux0(ie0) = b0(ie0)*(u(ie0)-u(ie0-1))/dx(0)
      enddo

      return
      end
c
      recursive subroutine cellpoissonflux1d(
     &  ifirst0,ilast0,
     &  dx,
     &  gcw,
     &  u,
     &  flux0)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL dx(0:NDIM-1)
      integer gcw
      REAL u(CELL1d(ifirst,ilast,gcw))
      REAL flux0(FACE1d(ifirst,ilast,0))
      integer ie0
c
c***********************************************************************
c
      do ie0 = ifirst0, ilast0+1
         flux0(ie0) = (u(ie0)-u(ie0-1))/dx(0)
      enddo

      return
      end
