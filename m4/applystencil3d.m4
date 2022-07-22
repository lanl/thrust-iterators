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
define(NDIM,3)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim3d.i)dnl

      recursive subroutine applystencilatpoint3d(
     &  lo0,hi0,
     &  lo1, hi1,
     &  lo2, hi2,
     &  i,j,k,
     &  stencilsize,
     &  offsets,
     &  stencil,
     &  a,
     &  b,
     &  fgcw,
     &  f, 
     &  ugcw,
     &  u, 
     &  rgcw,
     &  r)

      implicit none
c
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL3d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL3d(lo,hi,ugcw))
      REAL f(CELL3d(lo,hi,fgcw))
      REAL r(CELL3d(lo,hi,rgcw))
      integer i,j,k,s, io, jo,ko
c
c***********************************************************************
c
      r(i,j,k)=b*f(i,j,k)
      do s=1,stencilsize
         io=offsets(1,s)
         jo=offsets(2,s)
         ko=offsets(3,s)
         r(i,j,k)=r(i,j,k)+a*stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
      end do

      return
      end

      recursive subroutine applystencilonpatch3d(
     &  lo0,hi0,
     &  lo1, hi1,
     &  lo2, hi2,
     &  stencilsize,
     &  offsets,
     &  stencil,
     &  a,
     &  b,
     &  fgcw,
     &  f, 
     &  ugcw,
     &  u, 
     &  rgcw,
     &  r)

      implicit none
c
      integer lo0,lo1,lo2
      integer hi0,hi1,hi2
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL3d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL3d(lo,hi,ugcw))
      REAL f(CELL3d(lo,hi,fgcw))
      REAL r(CELL3d(lo,hi,rgcw))
      integer i,j,k,s, io, jo, ko
c
c***********************************************************************
c
      
      do k=lo2,hi2
         do j=lo1,hi1
            do i=lo0,hi0
               r(i,j,k)=b*f(i,j,k)
               do s=1,stencilsize
                  io=offsets(1,s)
                  jo=offsets(2,s)
                  ko=offsets(3,s)
                  r(i,j,k)=r(i,j,k)+a*stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
               end do
            end do
         end do
      enddo

      return
      end
