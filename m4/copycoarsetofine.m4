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

      recursive subroutine copycoarsetofine3d(
     & clo0, chi0, clo1, chi1, clo2, chi2,
     & flo0, fhi0, flo1, fhi1, flo2, fhi2,
     & axis,
     & cblo0, cbhi0, cblo1, cbhi1, cblo2, cbhi2,
     & fblo0, fbhi0, fblo1, fbhi1, fblo2, fbhi2,
     & gcw,
     & ratio,
     & cdata,
     & fdata )

      implicit none

c Coarse index space for the common edge.
      integer clo0,clo1,clo2,chi0,chi1,chi2

c Fine index space for the common edge.
      integer flo0,flo1,flo2,fhi0,fhi1,fhi2

C Coarse index space for the coarse data.
      integer cblo0,cblo1,cblo2,cbhi0,cbhi1,cbhi2

C Fine index space for the fine data.
      integer fblo0,fblo1,fblo2,fbhi0,fbhi1,fbhi2

      integer gcw

      integer ratio(0:2)

      integer axis

      double precision cdata(CELL3d(cblo,cbhi,gcw))
      double precision fdata(CELL3d(fblo,fbhi,gcw))

      integer fi,fj,fk,ci,cj,ck
      integer m

      if(axis.eq.X) then

         fj=flo1
         fk=flo2
         cj=clo1
         ck=clo2

c         do ci = clo0, chi0
c            do m = 0, ratio(0)-1
c               fdata(fi+m,fj,fk) = cdata(ci,cj,ck)
c            enddo
c            fi = fi+ratio(0)
c         enddo

c The coarse index computation is based on SAMRAI-overview.pdf.
c coarse = (fine+1)/ratio-1, lower < 0
c        = fine/ratio,       otherwise
c The index must be calculated based on the fine index
c instead of the way that are commented out.
c The reason is the commented method only works, for example,
c when flo is even when the refinement ratio is 2.
c When flo is odd, wrong values would be copied.

         do fi = flo0, fhi0
            if(fi.lt.0) then
               ci = (fi+1)/ratio(0)-1
            else
               ci = fi/ratio(0)
            endif
            fdata(fi,fj,fk) = cdata(ci,cj,ck)
         enddo

      elseif(axis.eq.Y) then

         fi=flo0
         fk=flo2
         ci=clo0
         ck=clo2

c         do cj = clo1, chi1
c            do m = 0, ratio(1)-1
c               fdata(fi,fj+m,fk) = cdata(ci,cj,ck)
c            enddo
c            fj = fj+ratio(1)
c         end do

         do fj = flo1, fhi1
            if(fj.lt.0) then
               cj = (fj+1)/ratio(1)-1
            else
               cj = fj/ratio(1)
            endif
            fdata(fi,fj,fk) = cdata(ci,cj,ck)
         enddo

      else if ( axis .eq. Z) then

         fi=flo0
         fj=flo1
         ci=clo0
         cj=clo1

c         do ck = clo2, chi2
c            do m = 0, ratio(2)-1
c               fdata(fi,fj,fk+m) = cdata(ci,cj,ck)
c            enddo
c            fk = fk+ratio(2)
c         end do

         do fk = flo2, fhi2
            if(fk.lt.0) then
               ck = (fk+1)/ratio(2)-1
            else
               ck = fk/ratio(2)
            endif
            fdata(fi,fj,fk) = cdata(ci,cj,ck)
         enddo

      endif

      return
      end

      recursive subroutine copycoarsetofinecorner3d(
     & clo0, chi0, clo1, chi1, clo2, chi2,
     & flo0, fhi0, flo1, fhi1, flo2, fhi2,
     & cblo0, cbhi0, cblo1, cbhi1, cblo2, cbhi2,
     & fblo0, fbhi0, fblo1, fbhi1, fblo2, fbhi2,
     & gcw,
     & cdata,
     & fdata )

      implicit none

      integer clo0,clo1,clo2,chi0,chi1,chi2

      integer flo0,flo1,flo2,fhi0,fhi1,fhi2

      integer cblo0,cblo1,cblo2,cbhi0,cbhi1,cbhi2

      integer fblo0,fblo1,fblo2,fbhi0,fbhi1,fbhi2

      integer gcw

      integer axis

      double precision cdata(CELL3d(cblo,cbhi,gcw))
      double precision fdata(CELL3d(fblo,fbhi,gcw))

      integer fi, fj, fk
      integer ic, jc, kc

      fi=flo0
      fj=flo1
      fk=flo2
      ic=clo0
      jc=clo1
      kc=clo2

      fdata(fi,fj,fk) = cdata(ic,jc,kc)

      return
      end


      recursive subroutine linearextrap(
     & lo0, hi0, lo1, hi1, lo2, hi2, gcw,
     & u,
     & bdrySegLo, bdrySegHi,
     & face, type)

c  Set homogeneous boundary conditions.
c  Will do linear extrapolation

      implicit none

      integer lo0
      integer lo1
      integer lo2

      integer hi0
      integer hi1
      integer hi2

      integer gcw

      integer face
      integer type

      integer bdrySegHi(0:NDIM-1)
      integer bdrySegLo(0:NDIM-1)

      double precision u(CELL3d(lo,hi,gcw))

      integer ihi
      integer ilo

      integer jhi
      integer jlo

      integer khi
      integer klo

      integer i
      integer j
      integer k

      if (type .eq. FACE) then
         if (face .eq. LEFT) then
            ilo = bdrySegLo(0)+1
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)

            do k = klo, khi
              do j = jlo, jhi
                 u(ilo-1,j,k) = 2.0*u(ilo,j,k)-u(ilo+1,j,k)
              end do
            enddo

         else if (face .eq. RIGHT) then
           ihi = bdrySegHi(0)-1
           jlo = bdrySegLo(1)
           jhi = bdrySegHi(1)
           klo = bdrySegLo(2)
           khi = bdrySegHi(2)

           do k = klo, khi
             do j = jlo, jhi
               u(ihi+1,j,k) = 2.0*u(ihi,j,k)-u(ihi-1,j,k)
             end do
           enddo

         else if (face .eq. BACK) then
            jlo = bdrySegLo(1)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)

            do k = klo, khi
              do i = ilo, ihi
                u(i,jlo-1,k) = 2.0*u(i,jlo,k)-u(i,jlo+1,k)
              enddo
            end do

         else if (face .eq. FRONT) then
            jhi = bdrySegHi(1)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            klo = bdrySegLo(2)
            khi = bdrySegHi(2)
            do k = klo, khi
              do i = ilo, ihi
                 u(i,jhi+1,k) = 2.0*u(i,jhi,k)-u(i,jhi-1,k)
              enddo
            end do

         else if (face .eq. BOTTOM) then
            klo = bdrySegLo(2)+1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            do j = jlo, jhi
              do i = ilo, ihi
                u(i,j,klo-1) = 2.0*u(i,j,klo)-u(i,j,klo+1)
              enddo
            end do

         else if (face .eq. TOP) then
            khi = bdrySegHi(2)-1
            ilo = bdrySegLo(0)
            ihi = bdrySegHi(0)
            jlo = bdrySegLo(1)
            jhi = bdrySegHi(1)
            do j = jlo, jhi
              do i = ilo, ihi
                 u(i,j,khi+1) = 2.0*u(i,j,khi)-u(i,j,khi-1)
              enddo
            end do

         else
            write (0,*) "SAMRUtils::copycoarsetorefine:extrapolate():
     &                   Error: unkown face."
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
