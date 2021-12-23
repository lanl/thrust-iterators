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
c  File:        solv_poissonresiduals1d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2000 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2531 $
c  Modified:    $Date: 2006-05-11 15:38:55 -0600 (Thu, 11 May 2006) $
c  Description: F77 routines that compute residual like quantities
c
define(NDIM,1)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim1d.i)dnl
c
c
      recursive subroutine nodediffusionv1apply1d(
     &  ifirst0,ilast0,
     &  alpha,beta,
     &  dx,
     &  agcw,
     &  acoef,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,
     &  a,
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL alpha,beta
      integer agcw, ugcw,fgcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  acoef(CELL1d(ifirst,ilast,agcw)),
     &  u(CELL1d(ifirst,ilast,ugcw)),
     &  f(CELL1d(ifirst,ilast,fgcw)),
     &  res(CELL1d(ifirst,ilast,rgcw))
      REAL flux0(FACE1d(ifirst,ilast,0))
      integer ic0
      REAL a
      REAL b
      REAL diver,lu
c
c***********************************************************************
c
      do ic0 = ifirst0, ilast0
         diver = beta*((flux0(ic0+1) - flux0(ic0))/dx(0))
         lu = - diver +alpha*acoef(ic0)*u(ic0)
         res(ic0) = b*f(ic0) +a*lu 
      enddo

      return
      end
c
      recursive subroutine nodediffusionv2apply1d(
     &  ifirst0,ilast0,
     &  alpha,beta,
     &  dx,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,
     &  a,
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL alpha,beta
      integer ugcw,fgcw,rgcw
      REAL dx(0:NDIM-1)
      REAL
     &  u(CELL1d(ifirst,ilast,ugcw)),
     &  f(CELL1d(ifirst,ilast,fgcw)),
     &  res(CELL1d(ifirst,ilast,rgcw))
      REAL flux0(FACE1d(ifirst,ilast,0))
      integer ic0
      REAL a
      REAL b
      REAL diver,lu
c
c***********************************************************************
c
      do ic0 = ifirst0, ilast0
         diver = beta*((flux0(ic0+1) - flux0(ic0))/dx(0))
         lu = - diver +alpha*u(ic0)
         res(ic0) = b*f(ic0) +a*lu 
      enddo

      return
      end
c
      recursive subroutine nodepoissonapply1d(
     &  ifirst0,ilast0,
     &  beta,
     &  dx,
     &  fgcw,
     &  f,
     &  flux0,
     &  a,  
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL beta
      REAL dx(0:NDIM-1)
      integer fgcw, rgcw
      REAL
     &  f(CELL1d(ifirst,ilast,fgcw)),
     &  res(CELL1d(ifirst,ilast,rgcw))
      REAL flux0(FACE1d(ifirst,ilast,0))
      integer ic0
      REAL a
      REAL b
      REAL lu
c
c***********************************************************************
c
      do ic0 = ifirst0, ilast0
         lu = -beta*((flux0(ic0+1) - flux0(ic0))/dx(0))
         res(ic0) = b*f(ic0) + a*lu 
      enddo

      return
      end
