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
c  File:        celldiffusionstencilcoeffs3d.m4
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
      recursive subroutine adjcellcrsfinebdryrhs3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  stencilsize,
     &  offset,
     &  stencil,
     &  gcw,
     &  u, rhs)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer pfirst0,pfirst1,pfirst2
      integer plast0,plast1,plast2
      integer direction, side
      integer stencilsize
      integer offset(0:NDIM-1,0:stencilsize-1)
      REAL stencil(0:stencilsize-1,CELL3d(ifirst,ilast,0))
      integer gcw
      REAL u(CELL3d(ifirst,ilast,gcw))
      REAL rhs(CELL3d(ifirst,ilast,0))
      integer ic0,ic1,ic2
      integer ie0,ie1,ie2
      integer i,j,k,s
      REAL ul(-1:1,-1:1,-1:1)
c
c***********************************************************************
c
c for now we assume that the maximum stencil size is 27
c ie there are atmost twenty seven non zero stencil entries and the stencil
c offsets are compact, ie at most offset by (+-1,+-1)
c ul represents the local u values in the direction we are interested
c in adjusting the stencil

      do k=-1,1
         do j=-1,1
            do i=-1,1
               ul(i,j,k)=0.0
            end do
         end do
      end do

      if (direction.eq.0) then
         ie0 = pfirst0+1-(2*side)
         if(side.eq.0) then
            do ic2 = pfirst2, plast2
               do ic1 = pfirst1, plast1
                  i=-1
                  do k=-1,1
                     do j=-1,1
                        ul(i,j,k)=u(ie0+i,ic1+j,ic2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ie0,ic1,ic2) = rhs(ie0,ic1,ic2) -
     &                                  stencil(s,ie0,ic1,ic2)*ul(i,j,k)
                  end do
               enddo
            enddo
         else
            do ic2 = pfirst2, plast2
               do ic1 = pfirst1, plast1
                  i=1
                  do k=-1,1
                     do j=-1,1
                        ul(i,j,k)=u(ie0+i,ic1+j,ic2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ie0,ic1,ic2) = rhs(ie0,ic1,ic2) -
     &                                  stencil(s,ie0,ic1,ic2)*ul(i,j,k)
                  end do
               enddo
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1+1-(2*side)
         if(side.eq.0) then
            do ic2 = pfirst2, plast2
               do ic0 = pfirst0, plast0
                  j=-1
                  do k=-1,1
                     do i=-1,1
                        ul(i,j,k)=u(ic0+i,ie1+j,ic2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ic0,ie1,ic2) = rhs(ic0,ie1,ic2)-
     &                                  stencil(s,ic0,ie1,ic2)*ul(i,j,k)
                  end do
               enddo
            enddo
         else
            do ic2 = pfirst2, plast2
               do ic0 = pfirst0, plast0
                  j=1
                  do k=-1,1
                     do i=-1,1
                        ul(i,j,k)=u(ic0+i,ie1+j,ic2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ic0,ie1,ic2) = rhs(ic0,ie1,ic2)-
     &                                  stencil(s,ic0,ie1,ic2)*ul(i,j,k)
                  end do
               enddo
            enddo
         endif
      elseif (direction.eq.2) then
         ie2 = pfirst2+1-(2*side)
         if(side.eq.0) then
            do ic1 = pfirst1, plast1
               do ic0 = pfirst0, plast0
                  k=-1
                  do j=-1,1
                     do i=-1,1
                        ul(i,j,k)=u(ic0+i,ic1+j,ie2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ic0,ic1,ie2) = rhs(ic0,ic1,ie2)-
     &                                  stencil(s,ic0,ic1,ie2)*ul(i,j,k)
                  end do
               enddo
            enddo
         else
            do ic1 = pfirst1, plast1
               do ic0 = pfirst0, plast0
                  k=1
                  do j=-1,1
                     do i=-1,1
                        ul(i,j,k)=u(ic0+i,ic1+j,ie2+k)
                     end do
                  end do
                  do s=0,stencilsize-1
                     i=offset(0,s)
                     j=offset(1,s)
                     k=offset(2,s)
                     rhs(ic0,ic1,ie2) = rhs(ic0,ic1,ie2)-
     &                                  stencil(s,ic0,ic1,ie2)*ul(i,j,k)
                  end do
               enddo
            enddo
         endif
      endif
      return
      end
