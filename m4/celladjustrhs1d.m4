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
c  File:        celldiffusionstencilcoeffs1d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2533 $
c  Modified:    $Date: 2006-05-11 15:57:25 -0600 (Thu, 11 May 2006) $
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
      recursive subroutine adjcellcrsfinebdryrhs1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  stencilsize,
     &  offset,
     &  stencil,
     &  gcw,
     &  u, rhs)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL1d(ifirst,ilast,0))
      integer gcw
      REAL
     &  u(CELL1d(ifirst,ilast,gcw)),
     &  rhs(CELL1d(ifirst,ilast,0))
      integer ie0
      integer i,s
      REAL ul(-1:1)
c
c***********************************************************************
c
c for now we assume that the maximum stencil size is 3
c ie there are atmost three non zero stencil entries and the stencil
c offsets are compact, ie at most offset by (+-1,+-1)
c ul represents the local u values in the direction we are interested
c in adjusting the stencil

      do i=-1,1
            ul(i)=0.0
      end do

      ie0 = pfirst0+1-(2*side)
      if(side.eq.0) then            
         i=-1
         ul(i)=u(ie0+i)
         do s=0,stencilsize-1
            i=offset(0,s)
            rhs(ie0) = rhs(ie0)-stencil(s,ie0)*ul(i)
         end do
      else
         i=1
         ul(i)=u(ie0+i)
         do s=0,stencilsize-1
            i=offset(0,s)
            rhs(ie0) = rhs(ie0)-stencil(s,ie0)*ul(i)
         end do
      endif

      return
      end

      recursive subroutine celladjustsystemrhspatch1d(
     &  ifirst0,ilast0,
     &  noffsets,
     &  offset,
     &  stencilsize,
     &  stencil,
     &  ugcw,
     &  u, 
     &  fgcw,
     &  f)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      integer noffsets
      integer stencilsize
      integer offset(1:NDIM,1:noffsets)
      REAL stencil(1:stencilsize,CELL1d(ifirst,ilast,0))
      integer ugcw, fgcw
      REAL u(CELL1d(ifirst,ilast,ugcw))
      REAL f(CELL1d(ifirst,ilast,fgcw))
      integer i,s, io
c
c***********************************************************************
c
c for now we assume that the maximum stencil size is 3
c ie there are atmost three non zero stencil entries and the stencil
c offsets are compact, ie at most offset by (+-1,+-1)
c ul represents the local u values in the direction we are interested
c in adjusting the stencil

      do i=ifirst0,ilast0
         do s=1,noffsets
            io=offset(1,s)
            f(i)=f(i)+stencil(s,i)*u(i+io)
         end do
      end do

      return
      end
