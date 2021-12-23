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
include(pdat_m4arrdim2d.i)dnl

define(EDGE, 1)
define(NODE, 2)

define(LEFT,   0)
define(RIGHT,  1)
define(BOTTOM, 2)
define(TOP,    3)

define(LL, 0)
define(LR, 1)
define(UL, 2)
define(UR, 3)

define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)

      recursive subroutine cellsetcorrectionbc2d(
     & lo0, hi0, lo1, hi1,
     & dx,
     & dgcw,
     & d0,d1,
     & ugcw,
     & u,
     & bdrySegLo, bdrySegHi,
     & extrapOrder,
     & face, type,
     & bdryType,
     & alpha,
     & beta )

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0
      integer lo1

      integer hi0
      integer hi1

      integer dgcw
      integer ugcw

      integer extrapOrder
      integer face
      integer type
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision d0(FACE2d0(lo,hi,dgcw))
      double precision d1(FACE2d1(lo,hi,dgcw))
      double precision u(CELL2d(lo,hi,ugcw))

      integer i
      integer ihi
      integer ilo
      integer j
      integer jhi
      integer jlo

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      double precision alpha
      double precision beta

      parameter      ( zero=0.0d0 )

      if (type .eq. EDGE) then
         if (face .eq. LEFT) then
            ilo = bdrySegLo(0)+1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 0) then
                     u(ilo-1,j) = 0.0
                  else if( extrapOrder .eq. 1) then
                     u(ilo-1,j) = -u(ilo,j)
                  else if ( extrapOrder .eq. 2) then
                     u(ilo-1,j) = -2.0*u(ilo,j)+u(ilo+1,j)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do j = jlo, jhi
                  u(ilo-1,j) = u(ilo,j)
               end do
            else if ( bdryType .eq. ROBIN ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                     factor=(2.0d0*alpha*d0(ilo,j)-h*beta)
                     factor=factor/(2.0d0*alpha*d0(ilo,j)+h*beta)
                    u(ilo-1,j)=factor*u(ilo,j)
                  else if ( extrapOrder .eq. 2) then
                     b=d0(ilo,j)
                     coeff(1) = 8.0d0*alpha*b-6.0d0*beta*h
                     coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                     coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                     u(ilo-1,j)=coeff(1)*u(ilo,j)+coeff(2)*u(ilo+1,j)
                  endif
               end do
            end if

         else if (face .eq. RIGHT) then

            ihi = bdrySegHi(0)-1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 0) then
                     u(ihi+1,j) = 0.0 
                  else if( extrapOrder .eq. 1) then
                     u(ihi+1,j) = -u(ihi,j)
                  else if ( extrapOrder .eq. 2) then
                     u(ihi+1,j) = -2.0*u(ihi,j)+u(ihi-1,j)/3.0
                  endif
              end do
            else if ( bdryType .eq. NEUMANN ) then
               do j = jlo, jhi
                  u(ihi+1,j) = u(ihi,j)
               end do
            else if ( bdryType .eq. ROBIN ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                     factor=(2.0d0*alpha*d0(ihi+1,j)-h*beta)
                     factor=factor/(2.0d0*alpha*d0(ihi+1,j)+h*beta)
                     u(ihi+1,j)=factor*u(ihi,j)
                  else if ( extrapOrder .eq. 2) then
                     b=d0(ihi+1,j)
                     coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                     coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                     coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                     u(ihi+1,j)=coeff(1)*u(ihi,j)+coeff(2)*u(ihi-1,j)
                  endif
               end do
            end if

         else if (face .eq. BOTTOM) then

            jlo = bdrySegLo(1)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
               do i = ilo, ihi
                  if( extrapOrder .eq. 0) then
                     u(i,jlo-1) = 0.0
                  else if( extrapOrder .eq. 1) then
                     u(i,jlo-1) = -u(i,jlo)
                  else if ( extrapOrder .eq. 2) then
                     u(i,jlo-1) = -2.0*u(i,jlo)+u(i,jlo+1)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do i = ilo, ihi
                  u(i,jlo-1) = u(i,jlo)
               end do
            else if ( bdryType .eq. ROBIN ) then
            end if

         else if (face .eq. TOP) then

            jhi = bdrySegHi(1)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
               do i = ilo, ihi
                  if( extrapOrder .eq. 0) then
                      u(i,jhi+1) = 0.0
                  else if( extrapOrder .eq. 1) then
                     u(i,jhi+1) = -u(i,jhi)
                  else if ( extrapOrder .eq. 2) then
                     u(i,jhi+1) = -2.0*u(i,jhi)+u(i,jhi-1)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do i = ilo, ihi
                  u(i,jhi+1) = u(i,jhi)
               end do
            else if ( bdryType .eq. ROBIN ) then
            end if

         end if

      else if (type .eq. NODE) then

c Corner values are set so that results of bilinear interpolation to 
c the next finer level that depend on the corner value are the same 
c as would have been obtained using barycentric interpolation that
c ignored the corner value.

         if (face .eq. LL) then

            ilo = bdrySegLo(0)
            jlo = bdrySegLo(1)

            u(ilo,jlo) = - u(ilo+1,jlo+1) + u(ilo,jlo+1) + u(ilo+1,jlo)

         else if (face .eq. LR) then

            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)

            u(ihi,jlo) = - u(ihi-1,jlo+1) + u(ihi,jlo+1) + u(ihi-1,jlo)

         else if (face .eq. UL) then

            ilo = bdrySegLo(0)
            jhi = bdrySegHi(1)

            u(ilo,jhi) = - u(ilo+1,jhi-1) + u(ilo,jhi-1) + u(ilo+1,jhi)

         else if (face .eq. UR) then

            ihi = bdrySegHi(0)
            jhi = bdrySegHi(1)

            u(ihi,jhi) = - u(ihi-1,jhi-1) + u(ihi,jhi-1) + u(ihi-1,jhi)

         endif

      end if

      return 
      end

      recursive subroutine cellsetpoissoncorrectionbc2d(
     & lo0, hi0, lo1, hi1,
     & dx,
     & ugcw,
     & u,
     & bdrySegLo, bdrySegHi,
     & extrapOrder,
     & face, type,
     & bdryType,
     & alpha,
     & beta  )

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0
      integer lo1

      integer hi0
      integer hi1

      integer ugcw

      integer extrapOrder
      integer face
      integer type
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision u(CELL2d(lo,hi,ugcw))

      integer i
      integer ihi
      integer ilo
      integer j
      integer jhi
      integer jlo

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      double precision alpha
      double precision beta

      parameter      ( zero=0.0d0 )

      if (type .eq. EDGE) then
         if (face .eq. LEFT) then
            ilo = bdrySegLo(0)+1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                     u(ilo-1,j) = -u(ilo,j)
                  else if ( extrapOrder .eq. 2) then
                     u(ilo-1,j) = -2.0*u(ilo,j)+u(ilo+1,j)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do j = jlo, jhi
                  u(ilo-1,j) = u(ilo,j)
               end do
            else if ( bdryType .eq. ROBIN ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                    factor=(2.0d0*alpha-h*beta)/(2.0d0*alpha+h*beta)
                    u(ilo-1,j)=factor*u(ilo,j)
                  else if ( extrapOrder .eq. 2) then
                     b=1.0
                     coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                     coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                     coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                     u(ilo-1,j)=coeff(1)*u(ilo,j)+coeff(2)*u(ilo+1,j)
                  endif
               end do
            end if

         else if (face .eq. RIGHT) then

            ihi = bdrySegHi(0)-1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                     u(ihi+1,j) = -u(ihi,j)
                  else if ( extrapOrder .eq. 2) then
                     u(ihi+1,j) = -2.0*u(ihi,j)+u(ihi-1,j)/3.0
                  endif
              end do
            else if ( bdryType .eq. NEUMANN ) then
               do j = jlo, jhi
                  u(ihi+1,j) = u(ihi,j)
               end do
            else if ( bdryType .eq. ROBIN ) then
               do j = jlo, jhi
                  if( extrapOrder .eq. 1) then
                     factor=(2.0d0*alpha-h*beta)/(2.0d0*alpha+h*beta)
                     u(ihi+1,j)=factor*u(ihi,j)
                  else if ( extrapOrder .eq. 2) then
                     b=1.0
                     coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                     coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                     coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                     u(ihi+1,j)=coeff(1)*u(ihi,j)+coeff(2)*u(ihi-1,j)
                  endif
               end do
            end if

         else if (face .eq. BOTTOM) then

            jlo = bdrySegLo(1)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
               do i = ilo, ihi
                  if( extrapOrder .eq. 1) then
                     u(i,jlo-1) = -u(i,jlo)
                  else if ( extrapOrder .eq. 2) then
                     u(i,jlo-1) = -2.0*u(i,jlo)+u(i,jlo+1)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do i = ilo, ihi
                  u(i,jlo-1) = u(i,jlo)
               end do
            else if ( bdryType .eq. ROBIN ) then
            end if

         else if (face .eq. TOP) then

            jhi = bdrySegHi(1)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
               do i = ilo, ihi
                  if( extrapOrder .eq. 1) then
                     u(i,jhi+1) = -u(i,jhi)
                  else if ( extrapOrder .eq. 2) then
                     u(i,jhi+1) = -2.0*u(i,jhi)+u(i,jhi-1)/3.0
                  endif
               end do
            else if ( bdryType .eq. NEUMANN ) then
               do i = ilo, ihi
                  u(i,jhi+1) = u(i,jhi)
               end do
            else if ( bdryType .eq. ROBIN ) then
            end if

         end if

      else if (type .eq. NODE) then

c Corner values are set so that results of bilinear interpolation to 
c the next finer level that depend on the corner value are the same 
c as would have been obtained using barycentric interpolation that
c ignored the corner value.

         if (face .eq. LL) then

            ilo = bdrySegLo(0)
            jlo = bdrySegLo(1)

            u(ilo,jlo) = - u(ilo+1,jlo+1) + u(ilo,jlo+1) + u(ilo+1,jlo)

         else if (face .eq. LR) then

            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)

            u(ihi,jlo) = - u(ihi-1,jlo+1) + u(ihi,jlo+1) + u(ihi-1,jlo)

         else if (face .eq. UL) then

            ilo = bdrySegLo(0)
            jhi = bdrySegHi(1)

            u(ilo,jhi) = - u(ilo+1,jhi-1) + u(ilo,jhi-1) + u(ilo+1,jhi)

         else if (face .eq. UR) then

            ihi = bdrySegHi(0)
            jhi = bdrySegHi(1)

            u(ihi,jhi) = - u(ihi-1,jhi-1) + u(ihi,jhi-1) + u(ihi-1,jhi)

         endif

      end if

      return 
      end

      recursive subroutine cellsetinteriorcornerbc2d(
     & lo0, hi0, lo1, hi1, gcw,
     & dx,
     & d0,d1,
     & u,
     & bdrySegLo, bdrySegHi,
     & extrapOrder,
     & face, type,
     & bdryType )

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0
      integer lo1

      integer hi0
      integer hi1

      integer gcw

      integer extrapOrder
      integer face
      integer type
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision d0(FACE2d0(lo,hi,0))
      double precision d1(FACE2d1(lo,hi,0))
      double precision u(CELL2d(lo,hi,gcw))

      integer ihi
      integer ilo
      integer jhi
      integer jlo

      double precision zero

      parameter      ( zero=0.0d0 )


c Corner values are set so that results of bilinear interpolation to 
c the next finer level that depend on the corner value are the same 
c as would have been obtained using barycentric interpolation that
c ignored the corner value.

      if (face .eq. LL) then
         
         ilo = bdrySegLo(0)
         jlo = bdrySegLo(1)
         
         u(ilo,jlo) = -5.0*u(ilo+1,jlo+1)
         
      else if (face .eq. LR) then
         
         ihi = bdrySegHi(0)
         jlo = bdrySegLo(1)
         
         u(ihi,jlo) = -5.0*u(ihi-1,jlo+1)
         
      else if (face .eq. UL) then
         
         ilo = bdrySegLo(0)
         jhi = bdrySegHi(1)
         
         u(ilo,jhi) = -5.0*u(ilo+1,jhi-1)
         
      else if (face .eq. UR) then
         
         ihi = bdrySegHi(0)
         jhi = bdrySegHi(1)
         
         u(ihi,jhi) = -5.0*u(ihi-1,jhi-1)
         
      endif
      
      return 
      end

      recursive subroutine cellsethomogenousbc2d(
     & lo0, lo1, hi0, hi1,
     & face,
     & segLo, segHi,
     & extrapOrder,
     & var )

c Set boundary conditions.  Currently this is only done on the north
c and south boundaries, where extrapolated values are stored.
c Homogeneous Dirichlet boundary values are assumed and ghost cells
c are filled using quadratic extrapolation normal to the boundary.

      implicit none

      integer lo0
      integer lo1

      integer hi0
      integer hi1

      integer face

      integer segHi(0:NDIM-1)
      integer segLo(0:NDIM-1)

      integer extrapOrder

      double precision var(CELL2d(lo,hi,1))

      double precision coeff(1:4)

      integer i,j
      integer ihi, jhi
      integer ilo, jlo
      integer iorder, jorder

      if ( extrapOrder .eq. 1 ) then
         coeff(1) = -1.0d0
      else if ( extrapOrder .eq. 2 ) then
         coeff(1) = -2.0d0
         coeff(2) = 1.0d0/3.0d0
      else if ( extrapOrder .eq. 3 ) then
         coeff(1) = -3.0d0
         coeff(2) = 1.0d0
         coeff(3) = -1.0d0/5.0d0
      else if ( extrapOrder .eq. 4 ) then
         coeff(1) = -4.0d0
         coeff(2) = 2.0d0
         coeff(3) = -4.0d0/5.0d0
         coeff(4) =  1.0d0/7.0d0
      endif

      if ( face .eq. 0 ) then
         ilo = lo0-1
         do j = max(lo1,segLo(1)), min(hi1,segHi(1))
            var(ilo,j) = 0.0d0
            do iorder=1,extrapOrder
               var(ilo,j) = var(ilo,j)+coeff(iorder)*var(ilo+iorder,j)
            end do
         end do
         var(ilo,lo1-1)=var(lo0, lo1)
         var(ilo,hi1+1)=var(lo0, hi1)
      else if ( face .eq. 1 ) then
         ihi = hi0+1
         do j = max(lo1,segLo(1)), min(hi1,segHi(1))
            var(ihi,j) = 0.0d0
            do iorder=1,extrapOrder
               var(ihi,j) = var(ihi,j)+coeff(iorder)*var(ihi-iorder,j)
            end do
         end do
         var(ihi,lo1-1)=var(hi0,lo1)
         var(ihi,hi1+1)=var(hi0,hi1)
      else if ( face .eq. 2 ) then
         jlo = lo1-1
         do i = max(lo0,segLo(0)), min(hi0,segHi(0))
            var(i,jlo) = 0.0d0
            do jorder=1,extrapOrder
               var(i,jlo) = var(i,jlo)+coeff(jorder)*var(i,jlo+jorder)
            end do
         end do
         var(lo0-1,jlo)=var(lo0,lo1)
         var(hi0+1,jlo)=var(hi0,lo1)
      else if ( face .eq. 3 ) then
         jhi = hi1+1
         do i = max(lo0,segLo(0)), min(hi0,segHi(0))
            var(i,jhi) = 0.0d0
            do jorder=1,extrapOrder
               var(i,jhi) = var(i,jhi)+coeff(jorder)*var(i,jhi-jorder)
            end do
         end do
         var(lo0-1,jhi)=var(lo0,hi1)
         var(hi0+1,jhi)=var(hi0,hi1)
      end if   

      return
      end
