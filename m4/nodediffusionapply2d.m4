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
c  Revision:    $Revision: 2489 $
c  Modified:    $Date: 2006-04-26 16:38:28 -0600 (Wed, 26 Apr 2006) $
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
      recursive subroutine nodediffusionv1apply2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  alpha,beta,
     &  dx,
     &  agcw,
     &  acoef,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,flux1,
     &  a,
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL alpha,beta
      integer agcw, ugcw,fgcw,rgcw
      REAL dx(0:1)
      REAL
     &  acoef(CELL2d(ifirst,ilast,agcw)),
     &  u(CELL2d(ifirst,ilast,ugcw)),
     &  f(CELL2d(ifirst,ilast,fgcw)),
     &  res(CELL2d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE2d0(ifirst,ilast,0)),
     &  flux1(FACE2d1(ifirst,ilast,0))
      integer ic0,ic1
      REAL a
      REAL b
      REAL diver,lu
c
c***********************************************************************
c
      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            diver = beta*((flux0(ic0+1,ic1) - flux0(ic0,ic1))/dx(0) +
     &                    (flux1(ic1+1,ic0) - flux1(ic1,ic0))/dx(1))
            lu = - diver +alpha*acoef(ic0,ic1)*u(ic0,ic1)
            res(ic0,ic1) = b*f(ic0,ic1) +a*lu 
         enddo
      enddo
      return
      end
c
      recursive subroutine nodediffusionv2apply2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  alpha,beta,
     &  dx,
     &  ugcw,
     &  u,
     &  fgcw,
     &  f,
     &  flux0,flux1,
     &  a,
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL alpha,beta
      integer ugcw,fgcw,rgcw
      REAL dx(0:1)
      REAL
     &  u(CELL2d(ifirst,ilast,ugcw)),
     &  f(CELL2d(ifirst,ilast,fgcw)),
     &  res(CELL2d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE2d0(ifirst,ilast,0)),
     &  flux1(FACE2d1(ifirst,ilast,0))
      integer ic0,ic1
      REAL a
      REAL b
      REAL diver,lu
c
c***********************************************************************
c
      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            diver = beta*((flux0(ic0+1,ic1) - flux0(ic0,ic1))/dx(0) +
     &                    (flux1(ic1+1,ic0) - flux1(ic1,ic0))/dx(1))
            lu = - diver +alpha*u(ic0,ic1)
            res(ic0,ic1) = b*f(ic0,ic1) +a*lu 
         enddo
      enddo
      return
      end
c
      recursive subroutine nodepoissonapply2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  beta,
     &  dx,
     &  fgcw,
     &  f,
     &  flux0,flux1, 
     &  a,  
     &  b,
     &  rgcw,
     &  res)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL beta
      REAL dx(0:1)
      integer fgcw, rgcw
      REAL
     &  f(CELL2d(ifirst,ilast,fgcw)),
     &  res(CELL2d(ifirst,ilast,rgcw))
      REAL
     &  flux0(FACE2d0(ifirst,ilast,0)),
     &  flux1(FACE2d1(ifirst,ilast,0))
      integer ic0,ic1
      REAL a
      REAL b
      REAL lu
c
c***********************************************************************
c
      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            lu = -beta*((flux0(ic0+1,ic1) - flux0(ic0,ic1))/dx(0) +
     &                  (flux1(ic1+1,ic0) - flux1(ic1,ic0))/dx(1))
            res(ic0,ic1) = b*f(ic0,ic1) + a*lu 
         enddo
      enddo
      return
      end
