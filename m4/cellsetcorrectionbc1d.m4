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
include(pdat_m4arrdim1d.i)dnl

define(LEFT,   0)
define(RIGHT,  1)

define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)

      recursive subroutine cellsetcorrectionbc1d(
     & lo0, hi0, 
     & dx,
     & dgcw,
     & d0,
     & ugcw,
     & u,
     & bdrySegLo, bdrySegHi,
     & extrapOrder,
     & face,
     & bdryType,
     & alpha,
     & beta  )

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0
      integer hi0

      integer dgcw
      integer ugcw

      integer extrapOrder
      integer face
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision d0(FACE1d(lo,hi,dgcw))
      double precision u(CELL1d(lo,hi,ugcw))

      double precision alpha
      double precision beta

      integer ihi
      integer ilo

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      parameter      ( zero=0.0d0 )

      if (face .eq. LEFT) then
         ilo = bdrySegLo(0)+1
         h=dx(0)
         if ( bdryType .eq. DIRICHLET ) then
            if( extrapOrder .eq. 1) then
               u(ilo-1) = -u(ilo)
            else if ( extrapOrder .eq. 2) then
               u(ilo-1) = -2.0*u(ilo)+u(ilo+1)/3.0
            endif
         else if ( bdryType .eq. NEUMANN ) then
            u(ilo-1) = u(ilo)
         else if ( bdryType .eq. ROBIN ) then
            if( extrapOrder .eq. 1) then
               factor=(4.0*d0(ilo)-h)/(4.0*d0(ilo)+h)
               u(ilo-1)=factor*u(ilo)
            else if ( extrapOrder .eq. 2) then
               b=d0(ilo)
               coeff(1)=(16.0*b-6.0*h)/(16.0*b+3.0*h)
               coeff(2)=h/(16.0*b+3.0*h)
               u(ilo-1)=coeff(1)*u(ilo)+coeff(2)*u(ilo+1)
            endif
         end if
         
      else if (face .eq. RIGHT) then
         
         ihi = bdrySegHi(0)-1
         h=dx(0)
         if ( bdryType .eq. DIRICHLET ) then
            if( extrapOrder .eq. 1) then
               u(ihi+1) = -u(ihi)
            else if ( extrapOrder .eq. 2) then
               u(ihi+1) = -2.0*u(ihi)+u(ihi-1)/3.0
            endif
         else if ( bdryType .eq. NEUMANN ) then
            u(ihi+1) = u(ihi)
         else if ( bdryType .eq. ROBIN ) then
            if( extrapOrder .eq. 1) then
               factor=(4.0*d0(ihi+1)-h)/(4.0*d0(ihi+1)+h)
               u(ihi+1)=factor*u(ihi)
            else if ( extrapOrder .eq. 2) then
               b=d0(ihi+1)
               coeff(1)=(16.0*b-6.0*h)/(3.0*h+16.0*b)
               coeff(2)=h/(3.0*h+16.0*b)
               u(ihi+1)=coeff(1)*u(ihi)+coeff(2)*u(ihi-1)
            endif
         end if
      end if

      return 
      end

      recursive subroutine cellsetpoissoncorrectionbc1d(
     & lo0, hi0,
     & dx,
     & ugcw,
     & u,
     & bdrySegLo, bdrySegHi,
     & extrapOrder,
     & face,
     & bdryType,
     & alpha,
     & beta )

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0

      integer hi0

      integer ugcw

      integer extrapOrder
      integer face
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision u(CELL1d(lo,hi,ugcw))

      integer ihi
      integer ilo

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      double precision alpha
      double precision beta

      parameter      ( zero=0.0d0 )

      if (face .eq. LEFT) then
         ilo = bdrySegLo(0)+1
         h=dx(0)
         if ( bdryType .eq. DIRICHLET ) then
            if( extrapOrder .eq. 1) then
               u(ilo-1) = -u(ilo)
            else if ( extrapOrder .eq. 2) then
               u(ilo-1) = -2.0*u(ilo)+u(ilo+1)/3.0
            endif
         else if ( bdryType .eq. NEUMANN ) then
            u(ilo-1) = u(ilo)
         else if ( bdryType .eq. ROBIN ) then
            if( extrapOrder .eq. 1) then
               factor=(4.0-h)/(4.0+h)
               u(ilo-1)=factor*u(ilo)
            else if ( extrapOrder .eq. 2) then
               b=1.0
               coeff(1)=(16.0*b-6.0*h)/(16.0*b+3.0*h)
               coeff(2)=h/(16.0*b+3.0*h)
               u(ilo-1)=coeff(1)*u(ilo)+coeff(2)*u(ilo+1)
            endif
         end if

      else if (face .eq. RIGHT) then

         ihi = bdrySegHi(0)-1
         h=dx(0)
         if ( bdryType .eq. DIRICHLET ) then
            if( extrapOrder .eq. 1) then
               u(ihi+1) = -u(ihi)
            else if ( extrapOrder .eq. 2) then
               u(ihi+1) = -2.0*u(ihi)+u(ihi-1)/3.0
            endif
         else if ( bdryType .eq. NEUMANN ) then
            u(ihi+1) = u(ihi)
         else if ( bdryType .eq. ROBIN ) then
            if( extrapOrder .eq. 1) then
               factor=(4.0-h)/(4.0+h)
               u(ihi+1)=factor*u(ihi)
            else if ( extrapOrder .eq. 2) then
               b=1.0
               coeff(1)=(16.0*b-6.0*h)/(3.0*h+16.0*b)
               coeff(2)=h/(3.0*h+16.0*b)
               u(ihi+1)=coeff(1)*u(ihi)+coeff(2)*u(ihi-1)
            endif
         end if
      end if

      return 
      end
