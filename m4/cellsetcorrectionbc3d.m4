C Copyright 2006, The Regents of the University
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
include(pdat_m4arrdim3d.i)dnl

define(FACE, 1)
define(EDGE, 2)
define(NODE, 3)

define(LEFT,   0)
define(RIGHT,  1)
define(BACK,   2)
define(FRONT,  3)
define(BOTTOM, 4)
define(TOP,    5)

define(LL, 0)
define(LR, 1)
define(UL, 2)
define(UR, 3)

define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)

define(X,0)
define(Y,1)
define(Z,2)

      recursive subroutine cellsetcorrectionbc3d(
     & lo0, hi0, lo1, hi1, lo2, hi2,
     & dx,
     & dgcw,
     & d0,d1,d2,
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
      integer lo2

      integer hi0
      integer hi1
      integer hi2

      integer dgcw
      integer ugcw

      integer extrapOrder
      integer face
      integer type
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision d0(FACE3d0(lo,hi,dgcw))
      double precision d1(FACE3d1(lo,hi,dgcw))
      double precision d2(FACE3d2(lo,hi,dgcw))
      double precision u(CELL3d(lo,hi,ugcw))

      integer ihi
      integer ilo

      integer jhi
      integer jlo

      integer khi
      integer klo

      integer i
      integer j
      integer k

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      double precision alpha
      double precision beta

      parameter      ( zero=0.0d0 )

      if (type .eq. FACE) then
         if (face .eq. LEFT) then
            ilo = bdrySegLo(0)+1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = max(lo2,klo), min(hi2,khi)
                  do j = max(lo1,jlo), min(hi1,jhi)
                        u(ilo-1,j,k) = -u(ilo,j,k)
                  end do
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = max(lo2,klo), min(hi2,khi)
                  do j = max(lo1,jlo), min(hi1,jhi)
                     u(ilo-1,j,k) = -2.0*u(ilo,j,k)+u(ilo+1,j,k)/3.0
                  end do
                enddo
              endif
            else if ( bdryType .eq. NEUMANN ) then
                do k = max(lo2,klo), min(hi2,khi)
                  do j = max(lo1,jlo), min(hi1,jhi)
                   u(ilo-1,j,k) = u(ilo,j,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
              if( extrapOrder .eq. 1) then
                do k = max(lo2,klo), min(hi2,khi)
                  do j = max(lo1,jlo), min(hi1,jhi)
                    factor=(2.0d0*alpha*d0(ilo,j,k)-h*beta)
                    factor=factor/(2.0d0*alpha*d0(ilo,j,k)+h*beta)
                    u(ilo-1,j,k)=factor*u(ilo,j,k)
                  end do
                end do
              else if ( extrapOrder .eq. 2) then
               do k = max(lo2,klo), min(hi2,khi)
                do j = max(lo1,jlo), min(hi1,jhi)
                 b=d0(ilo,j,k)
                 coeff(1) = 8.0d0*alpha*b-6.0d0*beta*h
                 coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                 coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                 u(ilo-1,j,k)=coeff(1)*u(ilo,j,k)+coeff(2)*u(ilo+1,j,k)
                end do
               end do
              endif
            endif

         else if (face .eq. RIGHT) then
           ihi = bdrySegHi(0)-1
           jlo = bdrySegLo(1)
           jhi = bdrySegHi(1)
           klo = bdrySegLo(2)
           khi = bdrySegHi(2)
           h=dx(0)
           if ( bdryType .eq. DIRICHLET ) then
             if( extrapOrder .eq. 1) then
               do k = max(lo2,klo), min(hi2,khi)
                 do j = max(lo1,jlo), min(hi1,jhi)
                   u(ihi+1,j,k) = -u(ihi,j,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do k = max(lo2,klo), min(hi2,khi)
                 do j = max(lo1,jlo), min(hi1,jhi)
                   u(ihi+1,j,k) = -2.0*u(ihi,j,k)+u(ihi-1,j,k)/3.0
                 end do
               enddo
             endif
           else if ( bdryType .eq. NEUMANN ) then
             do k = max(lo2,klo), min(hi2,khi)
               do j = max(lo1,jlo), min(hi1,jhi)
                 u(ihi+1,j,k) = u(ihi,j,k)
               end do
             enddo
           else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
              do k = max(lo2,klo), min(hi2,khi)
                 do j = max(lo1,jlo), min(hi1,jhi)
                   b=d0(ihi+1,j,k)
                   factor=(2.0d0*alpha*b-h*beta)
                   factor=factor/(2.0d0*alpha*b+h*beta)
                   u(ihi+1,j,k)=factor*u(ihi,j,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do k = max(lo2,klo), min(hi2,khi)
                 do j = max(lo1,jlo), min(hi1,jhi)
                  b=d0(ihi+1,j,k)
                  coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                  coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                  coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                  u(ihi+1,j,k)=coeff(1)*u(ihi,j,k)+coeff(2)*u(ihi-1,j,k)
                 end do
               enddo
             endif
           end if
         else if (face .eq. BACK) then
            jlo = bdrySegLo(1)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = max(lo2,klo), min(hi2,khi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                    u(i,jlo-1,k) = -u(i,jlo,k)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = max(lo2,klo), min(hi2,khi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                    u(i,jlo-1,k) = -2.0*u(i,jlo,k)+u(i,jlo+1,k)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do k = max(lo2,klo), min(hi2,khi)
                do i = max(lo0,ilo), min(hi0,ihi)
                  u(i,jlo-1,k) = u(i,jlo,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               do k = max(lo2,klo), min(hi2,khi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                   b=d1(jlo,k,i)
                   factor=(2.0d0*alpha*b-h*beta)
                   factor=factor/(2.0d0*alpha*b+h*beta)
                   u(i,jlo-1,k)=factor*u(i,jlo,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do k = max(lo2,klo), min(hi2,khi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                  b=d1(jlo,k,i)
                  coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                  coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                  coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                  u(i,jlo-1,k)=coeff(1)*u(i,jlo,k)+coeff(2)*u(i,jlo+1,k)
                 end do
               enddo
             endif
            end if
         else if (face .eq. FRONT) then
            jhi = bdrySegHi(1)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = max(lo2,klo), min(hi2,khi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                     u(i,jhi+1,k) = -u(i,jhi,k)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = max(lo2,klo), min(hi2,khi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                     u(i,jhi+1,k) = -2.0*u(i,jhi,k)+u(i,jhi-1,k)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do k = max(lo2,klo), min(hi2,khi)
                do i = max(lo0,ilo), min(hi0,ihi)
                  u(i,jhi+1,k) = u(i,jhi,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               do k = max(lo2,klo), min(hi2,khi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                   b=d1(jhi+1,k,i)
                   factor=(2.0d0*alpha*b-h*beta)
                   factor=factor/(2.0d0*alpha*b+h*beta)
                   u(i,jhi+1,k)=factor*u(i,jhi,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do k = max(lo2,klo), min(hi2,khi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                  b=d1(jhi+1,k,i)
                  coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                  coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                  coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                  u(i,jhi+1,k)=coeff(1)*u(i,jhi,k)+coeff(2)*u(i,jhi-1,k)
                 end do
               enddo
             endif
            end if
         else if (face .eq. BOTTOM) then
            klo = bdrySegLo(2)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(2)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do j = max(lo1,jlo), min(hi1,jhi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                    u(i,j,klo-1) = -u(i,j,klo)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do j = max(lo1,jlo), min(hi1,jhi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                    u(i,j,klo-1) = -2.0*u(i,j,klo)+u(i,j,klo+1)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do j = max(lo1,jlo), min(hi1,jhi)
                do i = max(lo0,ilo), min(hi0,ihi)
                  u(i,j,klo-1) = u(i,j,klo)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               do j = max(lo1,jlo), min(hi1,jhi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                   b=d2(klo,i,j)
                   factor=(2.0d0*alpha*b-h*beta)
                   factor=factor/(2.0d0*alpha*b+h*beta)
                   u(i,j,klo-1)=factor*u(i,j,klo)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do j = max(lo1,jlo), min(hi1,jhi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                  b=d2(klo,i,j)
                  coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                  coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                  coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                  u(i,j,klo-1)=coeff(1)*u(i,j,klo)+coeff(2)*u(i,j,klo+1)
                 end do
               enddo
             endif
            end if
         else if (face .eq. TOP) then
            khi = bdrySegHi(2)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(2)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
               do j = max(lo1,jlo), min(hi1,jhi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                     u(i,j,khi+1) = -u(i,j,khi)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do j = max(lo1,jlo), min(hi1,jhi)
                  do i = max(lo0,ilo), min(hi0,ihi)
                     u(i,j,khi+1) = -2.0*u(i,j,khi)+u(i,j,khi-1)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do j = max(lo1,jlo), min(hi1,jhi)
                do i = max(lo0,ilo), min(hi0,ihi)
                  u(i,j,khi+1) = u(i,j,khi)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               do j = max(lo1,jlo), min(hi1,jhi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                   b=d2(khi+1,i,j)
                   factor=(2.0d0*alpha*b-h*beta)
                   factor=factor/(2.0d0*alpha*b+h*beta)
                   u(i,j,khi+1)=factor*u(i,j,khi)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do j = max(lo1,jlo), min(hi1,jhi)
                 do i = max(lo0,ilo), min(hi0,ihi)
                  b=d2(khi+1,i,j)
                  coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
                  coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
                  coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
                  u(i,j,khi+1)=coeff(1)*u(i,j,khi)+coeff(2)*u(i,j,khi-1)
                 end do
               enddo
             endif
            end if
         end if

c Follow the logic for the 2D case.
c Corner values are set so that results of bilinear interpolation to
c the next finer level that depend on the corner value are the same
c as would have been obtained using barycentric interpolation that
c ignored the corner value.

      else if (type .eq. EDGE) then

        ilo = bdrySegLo(0)
        ihi = bdrySegHi(0)
        jlo = bdrySegLo(1)
        jhi = bdrySegHi(1)
        klo = bdrySegLo(2)
        khi = bdrySegHi(2)

c The variable face is actually the location index of the boundary box.
        if(face.eq.0) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo+1,jlo+1,k)
     &                     +u(ilo+1,jlo,k)+u(ilo,jlo+1,k)
          enddo
        else if(face.eq.1) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo-1,jlo+1,k)
     &                     +u(ilo-1,jlo,k)+u(ilo,jlo+1,k)
          enddo
        else if(face.eq.2) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo+1,jlo-1,k)
     &                     +u(ilo+1,jlo,k)+u(ilo,jlo-1,k)
          enddo
        else if(face.eq.3) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo-1,jlo-1,k)
     &                     +u(ilo-1,jlo,k)+u(ilo,jlo-1,k)
          enddo
        else if(face.eq.4) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo+1,j,klo+1)
     &                     +u(ilo+1,j,klo)+u(ilo,j,klo+1)
          enddo
        else if(face.eq.5) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo-1,j,klo+1)
     &                     +u(ilo-1,j,klo)+u(ilo,j,klo+1)
          enddo
        else if(face.eq.6) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo+1,j,klo-1)
     &                     +u(ilo+1,j,klo)+u(ilo,j,klo-1)
          enddo
        else if(face.eq.7) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo-1,j,klo-1)
     &                     +u(ilo-1,j,klo)+u(ilo,j,klo-1)
          enddo
        else if(face.eq.8) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo+1,klo+1)
     &                     +u(i,jlo+1,klo)+u(i,jlo,klo+1)
          enddo
        else if(face.eq.9) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo-1,klo+1)
     &                     +u(i,jlo-1,klo)+u(i,jlo,klo+1)
          enddo
        else if(face.eq.10) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo+1,klo-1)
     &                     +u(i,jlo+1,klo)+u(i,jlo,klo-1)
          enddo
        else if(face.eq.11) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo-1,klo-1)
     &                     +u(i,jlo-1,klo)+u(i,jlo,klo-1)
          enddo
        else
          write (0,*) "SAMRSolvers::cellsetcorrectionbc3d::
     &                 Unknown location index."
        endif

      else if (type .eq. NODE) then

        ilo = bdrySegLo(0)
        ihi = bdrySegHi(0)
        jlo = bdrySegLo(1)
        jhi = bdrySegHi(1)
        klo = bdrySegLo(2)
        khi = bdrySegHi(2)

        if(face.eq.0 .or. face.eq.4) then
          u(ilo,jlo,klo) = -u(ilo+1,jlo+1,klo)
     &                     +u(ilo+1,jlo,klo)+u(ilo,jlo+1,klo)
        else if(face.eq.1 .or. face.eq.5) then
          u(ilo,jlo,klo) = -u(ilo-1,jlo+1,klo)
     &                     +u(ilo-1,jlo,klo)+u(ilo,jlo+1,klo)
        else if(face.eq.2 .or. face.eq.6) then
          u(ilo,jlo,klo) = -u(ilo+1,jlo-1,klo)
     &                     +u(ilo+1,jlo,klo)+u(ilo,jlo-1,klo)
        else if(face.eq.3 .or. face.eq.7) then
          u(ilo,jlo,klo) = -u(ilo-1,jlo-1,klo)
     &                     +u(ilo-1,jlo,klo)+u(ilo,jlo-1,klo)
        else
          write (0,*) "SAMRSolvers::cellsetcorrectionbc3d::
     &                 Unknown location index."
        endif

      end if

      return
      end

      recursive subroutine cellsetpoissoncorrectionbc3d(
     & lo0, hi0, lo1, hi1, lo2, hi2,
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
      integer lo2

      integer hi0
      integer hi1
      integer hi2

      integer ugcw

      integer extrapOrder
      integer face
      integer type
      integer bdryType

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision dx(0:NDIM-1)

      double precision u(CELL3d(lo,hi,ugcw))

      integer ihi
      integer ilo

      integer jhi
      integer jlo

      integer khi
      integer klo

      integer i
      integer j
      integer k

      double precision factor
      double precision b, h
      double precision zero
      double precision coeff(1:2)

      double precision alpha
      double precision beta

      parameter      ( zero=0.0d0 )

      if (type .eq. FACE) then
         if (face .eq. LEFT) then
            ilo = bdrySegLo(0)+1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(0)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = klo, khi
                  do j = jlo, jhi
                        u(ilo-1,j,k) = -u(ilo,j,k)
                  end do
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = klo, khi
                  do j = jlo, jhi
                     u(ilo-1,j,k) = -2.0*u(ilo,j,k)+u(ilo+1,j,k)/3.0
                  end do
                enddo
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do k = klo, khi
                do j = jlo, jhi
                   u(ilo-1,j,k) = u(ilo,j,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
              if( extrapOrder .eq. 1) then
                factor=(2.0d0*alpha-h*beta)/(2.0d0*alpha+h*beta)
                do k = klo, khi
                  do j = jlo, jhi
                    u(ilo-1,j,k)=factor*u(ilo,j,k)
                  end do
                end do
              else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = 8.0d0*alpha*b-6.0d0*beta*h
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do k = klo, khi
                 do j = jlo, jhi
                  u(ilo-1,j,k)=coeff(1)*u(ilo,j,k)+coeff(2)*u(ilo+1,j,k)
                 end do
               end do
              endif
            endif

         else if (face .eq. RIGHT) then
           ihi = bdrySegHi(0)-1
           jlo = bdrySegLo(1)
           jhi = bdrySegHi(1)
           klo = bdrySegLo(2)
           khi = bdrySegHi(2)
           h=dx(0)
           if ( bdryType .eq. DIRICHLET ) then
             if( extrapOrder .eq. 1) then
               do k = klo, khi
                 do j = jlo, jhi
                   u(ihi+1,j,k) = -u(ihi,j,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               do k = klo, khi
                 do j = jlo, jhi
                   u(ihi+1,j,k) = -2.0*u(ihi,j,k)+u(ihi-1,j,k)/3.0
                 end do
               enddo
             endif
           else if ( bdryType .eq. NEUMANN ) then
             do k = klo, khi
               do j = jlo, jhi
                 u(ihi+1,j,k) = u(ihi,j,k)
               end do
             enddo
           else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               b=1.0d0
               factor=(2.0d0*alpha*b-h*beta)
               factor=factor/(2.0d0*alpha*b+h*beta)
               do k = klo, khi
                 do j = jlo, jhi
                   u(ihi+1,j,k)=factor*u(ihi,j,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do k = klo, khi
                 do j = jlo, jhi
                  u(ihi+1,j,k)=coeff(1)*u(ihi,j,k)+coeff(2)*u(ihi-1,j,k)
                 end do
               enddo
             endif
           end if
         else if (face .eq. BACK) then
            jlo = bdrySegLo(1)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = klo, khi
                  do i = ilo, ihi
                    u(i,jlo-1,k) = -u(i,jlo,k)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = klo, khi
                  do i = ilo, ihi
                    u(i,jlo-1,k) = -2.0*u(i,jlo,k)+u(i,jlo+1,k)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do k = klo, khi
                do i = ilo, ihi
                  u(i,jlo-1,k) = u(i,jlo,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
              if( extrapOrder .eq. 1) then
               b=1.0d0
               factor=(2.0d0*alpha*b-h*beta)
               factor=factor/(2.0d0*alpha*b+h*beta)
               do k = klo, khi
                 do i = ilo, ihi
                   u(i,jlo-1,k)=factor*u(i,jlo,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do k = klo, khi
                 do i = ilo, ihi
                  u(i,jlo-1,k)=coeff(1)*u(i,jlo,k)+coeff(2)*u(i,jlo+1,k)
                 end do
               enddo
             endif
            end if
         else if (face .eq. FRONT) then
            jhi = bdrySegHi(1)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            h=dx(1)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do k = klo, khi
                  do i = ilo, ihi
                     u(i,jhi+1,k) = -u(i,jhi,k)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do k = klo, khi
                  do i = ilo, ihi
                     u(i,jhi+1,k) = -2.0*u(i,jhi,k)+u(i,jhi-1,k)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do k = klo, khi
                do i = ilo, ihi
                  u(i,jhi+1,k) = u(i,jhi,k)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               b=1.0d0
               factor=(2.0d0*alpha*b-h*beta)
               factor=factor/(2.0d0*alpha*b+h*beta)
               do k = klo, khi
                 do i = ilo, ihi
                   u(i,jhi+1,k)=factor*u(i,jhi,k)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do k = klo, khi
                 do i = ilo, ihi
                  u(i,jhi+1,k)=coeff(1)*u(i,jhi,k)+coeff(2)*u(i,jhi-1,k)
                 end do
               enddo
             endif
            end if
         else if (face .eq. BOTTOM) then
            klo = bdrySegLo(2)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(2)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do j = jlo, jhi
                  do i = ilo, ihi
                    u(i,j,klo-1) = -u(i,j,klo)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do j = jlo, jhi
                  do i = ilo, ihi
                    u(i,j,klo-1) = -2.0*u(i,j,klo)+u(i,j,klo+1)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do j = jlo, jhi
                do i = ilo, ihi
                  u(i,j,klo-1) = u(i,j,klo)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               b=1.0d0
               factor=(2.0d0*alpha*b-h*beta)
               factor=factor/(2.0d0*alpha*b+h*beta)
               do j = jlo, jhi
                 do i = ilo, ihi
                   u(i,j,klo-1)=factor*u(i,j,klo)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do j = jlo, jhi
                 do i = ilo, ihi
                  u(i,j,klo-1)=coeff(1)*u(i,j,klo)+coeff(2)*u(i,j,klo+1)
                 end do
               enddo
             endif
            end if
         else if (face .eq. TOP) then
            khi = bdrySegHi(2)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            h=dx(2)
            if ( bdryType .eq. DIRICHLET ) then
              if( extrapOrder .eq. 1) then
                do j = jlo, jhi
                  do i = ilo, ihi
                     u(i,j,khi+1) = -u(i,j,khi)
                  enddo
                enddo
              else if ( extrapOrder .eq. 2) then
                do j = jlo, jhi
                  do i = ilo, ihi
                     u(i,j,khi+1) = -2.0*u(i,j,khi)+u(i,j,khi-1)/3.0
                  enddo
                end do
              endif
            else if ( bdryType .eq. NEUMANN ) then
              do j = jlo, jhi
                do i = ilo, ihi
                  u(i,j,khi+1) = u(i,j,khi)
                end do
              enddo
            else if ( bdryType .eq. ROBIN ) then
             if( extrapOrder .eq. 1) then
               b=1.0d0
               factor=(2.0d0*alpha*b-h*beta)
               factor=factor/(2.0d0*alpha*b+h*beta)
               do j = jlo, jhi
                 do i = ilo, ihi
                   u(i,j,khi+1)=factor*u(i,j,khi)
                 enddo
               enddo
             else if ( extrapOrder .eq. 2) then
               b=1.0d0
               coeff(1) = (8.0d0*alpha*b-6.0d0*beta*h)
               coeff(1) = coeff(1)/(8.0d0*alpha*b+3.0d0*beta*h)
               coeff(2) = h/(8.0d0*alpha*b+3.0d0*beta*h)
               do j = jlo, jhi
                 do i = ilo, ihi
                  u(i,j,khi+1)=coeff(1)*u(i,j,khi)+coeff(2)*u(i,j,khi-1)
                 end do
               enddo
             endif
            end if
         end if

      else if (type .eq. EDGE) then

        ilo = bdrySegLo(0)
        ihi = bdrySegHi(0)
        jlo = bdrySegLo(1)
        jhi = bdrySegHi(1)
        klo = bdrySegLo(2)
        khi = bdrySegHi(2)

c The variable face is actually the location index of the boundary box.
        if(face.eq.0) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo+1,jlo+1,k)
     &                     +u(ilo+1,jlo,k)+u(ilo,jlo+1,k)
          enddo
        else if(face.eq.1) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo-1,jlo+1,k)
     &                     +u(ilo-1,jlo,k)+u(ilo,jlo+1,k)
          enddo
        else if(face.eq.2) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo+1,jlo-1,k)
     &                     +u(ilo+1,jlo,k)+u(ilo,jlo-1,k)
          enddo
        else if(face.eq.3) then
          do k = klo,khi
            u(ilo,jlo,k) = -u(ilo-1,jlo-1,k)
     &                     +u(ilo-1,jlo,k)+u(ilo,jlo-1,k)
          enddo
        else if(face.eq.4) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo+1,j,klo+1)
     &                     +u(ilo+1,j,klo)+u(ilo,j,klo+1)
          enddo
        else if(face.eq.5) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo-1,j,klo+1)
     &                     +u(ilo-1,j,klo)+u(ilo,j,klo+1)
          enddo
        else if(face.eq.6) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo+1,j,klo-1)
     &                     +u(ilo+1,j,klo)+u(ilo,j,klo-1)
          enddo
        else if(face.eq.7) then
          do j = jlo,jhi
            u(ilo,j,klo) = -u(ilo-1,j,klo-1)
     &                     +u(ilo-1,j,klo)+u(ilo,j,klo-1)
          enddo
        else if(face.eq.8) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo+1,klo+1)
     &                     +u(i,jlo+1,klo)+u(i,jlo,klo+1)
          enddo
        else if(face.eq.9) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo-1,klo+1)
     &                     +u(i,jlo-1,klo)+u(i,jlo,klo+1)
          enddo
        else if(face.eq.10) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo+1,klo-1)
     &                     +u(i,jlo+1,klo)+u(i,jlo,klo-1)
          enddo
        else if(face.eq.11) then
          do i = ilo,ihi
            u(i,jlo,klo) = -u(i,jlo-1,klo-1)
     &                     +u(i,jlo-1,klo)+u(i,jlo,klo-1)
          enddo
        else
          write (0,*) "SAMRSolvers::cellsetcorrectionbc3d::
     &                 Unknown location index."
        endif

      else if (type .eq. NODE) then

        ilo = bdrySegLo(0)
        ihi = bdrySegHi(0)
        jlo = bdrySegLo(1)
        jhi = bdrySegHi(1)
        klo = bdrySegLo(2)
        khi = bdrySegHi(2)

        if(face.eq.0 .or. face.eq.4) then
          u(ilo,jlo,klo) = -u(ilo+1,jlo+1,klo)
     &                     +u(ilo+1,jlo,klo)+u(ilo,jlo+1,klo)
        else if(face.eq.1 .or. face.eq.5) then
          u(ilo,jlo,klo) = -u(ilo-1,jlo+1,klo)
     &                     +u(ilo-1,jlo,klo)+u(ilo,jlo+1,klo)
        else if(face.eq.2 .or. face.eq.6) then
          u(ilo,jlo,klo) = -u(ilo+1,jlo-1,klo)
     &                     +u(ilo+1,jlo,klo)+u(ilo,jlo-1,klo)
        else if(face.eq.3 .or. face.eq.7) then
          u(ilo,jlo,klo) = -u(ilo-1,jlo-1,klo)
     &                     +u(ilo-1,jlo,klo)+u(ilo,jlo-1,klo)
        else
          write (0,*) "SAMRSolvers::cellsetcorrectionbc3d::
     &                 Unknown location index."
        endif

      end if

      return
      end

