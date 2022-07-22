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
c  File:        celldiffusionstencilcoeffs2d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2057 $
c  Modified:    $Date: 2005-11-09 17:22:29 -0700 (Wed, 09 Nov 2005) $
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
      recursive subroutine adjcellcrsfinebdryrhs2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  stencilsize,
     &  offset,
     &  stencil,
     &  gcw,
     &  u, rhs)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL2d(ifirst,ilast,0))
      integer gcw
      REAL
     &  u(CELL2d(ifirst,ilast,gcw)),
     &  rhs(CELL2d(ifirst,ilast,0))
      integer ic0,ic1,ie0,ie1
      integer i,j,s
      REAL ul(-1:1,-1:1)
c
c***********************************************************************
c
c for now we assume that the maximum stencil size is 9
c ie there are atmost nine non zero stencil entries and the stencil
c offsets are compact, ie at most offset by (+-1,+-1)
c ul represents the local u values in the direction we are interested
c in adjusting the stencil

      do i=-1,1
         do j=-1,1
            ul(i,j)=0.0
         end do
      end do

      if (direction.eq.0) then
         ie0 = pfirst0+1-(2*side)
         if(side.eq.0) then            
            do ic1 = pfirst1, plast1
               i=-1
               do j=-1,1
                  ul(i,j)=u(ie0+i,ic1+j)
               end do
               do s=0,stencilsize-1
                  i=offset(0,s)
                  j=offset(1,s)
                  rhs(ie0,ic1) = rhs(ie0,ic1) -
     &              stencil(s,ie0,ic1)*ul(i,j)
               end do
            enddo
         else
            do ic1 = pfirst1, plast1
               i=1
               do j=-1,1
                  ul(i,j)=u(ie0+i,ic1+j)
               end do
               do s=0,stencilsize-1
                  i=offset(0,s)
                  j=offset(1,s)
                  rhs(ie0,ic1) = rhs(ie0,ic1) -
     &                 stencil(s,ie0,ic1)*ul(i,j)
                  end do
            end do
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1+1-(2*side)
         if(side.eq.0) then
            do ic0 = pfirst0, plast0
               j=-1
               do i=-1,1
                  ul(i,j)=u(ic0+i,ie1+j)
               end do
               do s=0,stencilsize-1
                  i=offset(0,s)
                  j=offset(1,s)
                  rhs(ic0,ie1) = rhs(ic0,ie1)-
     &                 stencil(s,ic0,ie1)*ul(i,j)
               end do
            enddo
         else
            do ic0 = pfirst0, plast0
               j=1
               do i=-1,1
                  ul(i,j)=u(ic0+i,ie1+j)
               end do
               do s=0,stencilsize-1
                  i=offset(0,s)
                  j=offset(1,s)
                  rhs(ic0,ie1) = rhs(ic0,ie1)-
     &                 stencil(s,ic0,ie1)*ul(i,j)
               end do
            enddo
         endif
      endif
      return
      end

      recursive subroutine celladjustsystemrhspatch2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
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
      integer ifirst0,ifirst1,ilast0,ilast1
      integer noffsets
      integer stencilsize
      integer offset(1:NDIM,1:noffsets)
      REAL stencil(1:stencilsize,CELL2d(ifirst,ilast,0))
      integer ugcw, fgcw
      REAL u(CELL2d(ifirst,ilast,ugcw))
      REAL f(CELL2d(ifirst,ilast,fgcw))
      integer i,j,s, io, jo
c
c***********************************************************************
c
c for now we assume that the maximum stencil size is 9
c ie there are atmost nine non zero stencil entries and the stencil
c offsets are compact, ie at most offset by (+-1,+-1)
c ul represents the local u values in the direction we are interested
c in adjusting the stencil

      do j=ifirst0,ilast0
         do i=ifirst1,ilast1
            do s=1,noffsets
               io=offset(1,s)
               jo=offset(2,s)
               f(i,j)=f(i,j)+stencil(s,i,j)*u(i+io,j+jo)
            end do
         end do
      end do

      return
      end
