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
define(NDIM,2)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim2d.i)dnl

      recursive subroutine applystencilatpoint2d(
     &  lo0,hi0,lo1, hi1,
     &  i,j,
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
      integer lo0,lo1,hi0,hi1
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL2d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL2d(lo,hi,ugcw))
      REAL f(CELL2d(lo,hi,fgcw))
      REAL r(CELL2d(lo,hi,rgcw))
      integer i,j,s, io, jo
c
c***********************************************************************
c
      r(i,j)=b*f(i,j)
      do s=1,stencilsize
         io=offsets(1,s)
         jo=offsets(2,s)
         r(i,j)=r(i,j)+a*stencil(s,i,j)*u(i+io,j+jo)
      end do

      return
      end

      recursive subroutine applystencilonpatch2d(
     &  lo0,hi0,lo1, hi1,
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
      integer lo0,lo1,hi0,hi1
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL2d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL2d(lo,hi,ugcw))
      REAL f(CELL2d(lo,hi,fgcw))
      REAL r(CELL2d(lo,hi,rgcw))
      integer i,j,s, io, jo
c
c***********************************************************************
c
      
      do j=lo0,hi0
         do i=lo1,hi1
            r(i,j)=b*f(i,j)
            do s=1,stencilsize
               io=offsets(1,s)
               jo=offsets(2,s)
               r(i,j)=r(i,j)+a*stencil(s,i,j)*u(i+io,j+jo)
            end do
         end do
      end do
      return
      end
