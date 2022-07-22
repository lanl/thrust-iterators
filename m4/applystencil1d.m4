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
define(NDIM,1)dnl
define(REAL,`double precision')dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim1d.i)dnl

      recursive subroutine applystencilatpoint1d(
     &  lo0,hi0,
     &  i,
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
      integer lo0,hi0
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL1d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL1d(lo,hi,ugcw))
      REAL f(CELL1d(lo,hi,fgcw))
      REAL r(CELL1d(lo,hi,rgcw))
      integer i,s, io
c
c***********************************************************************
c
      r(i)=b*f(i)
      do s=1,stencilsize
         io=offsets(1,s)
         r(i)=r(i)+a*stencil(s,i)*u(i+io)
      end do

      return
      end

      recursive subroutine applystencilonpatch1d(
     &  lo0,hi0,
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
      integer lo0,hi0
      integer stencilsize
      integer offsets(1:NDIM,1:stencilsize)
      REAL a,b
      REAL stencil(1:stencilsize,CELL1d(lo,hi,0))
      integer ugcw, fgcw,rgcw
      REAL u(CELL1d(lo,hi,ugcw))
      REAL f(CELL1d(lo,hi,fgcw))
      REAL r(CELL1d(lo,hi,rgcw))
      integer i,s, io
c
c***********************************************************************
c
      
      do i=lo0,hi0
         r(i)=b*f(i)
         do s=1,stencilsize
            io=offsets(1,s)
            r(i)=r(i)+a*stencil(s,i)*u(i+io)
         end do
      end do
      return
      end
