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

c
c  File:        celldiffusionstencilcoeffs3d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2886 $
c  Modified:    $Date: 2006-09-21 17:31:31 -0600 (Thu, 21 Sep 2006) $
c  Description: F77 routines that compute matrix entries for 3d cell centered diffusion solver.
c
define(NDIM,3)dnl
define(REAL,`double precision')dnl
define(PP,0)dnl
define(WW,1)dnl
define(EE,2)dnl
define(SS,3)dnl
define(NN,4)dnl
define(BB,5)dnl
define(TT,6)dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim3d.i)dnl
c
c
      recursive subroutine celldiffusionoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  bilo0,bilo1,bilo2,
     &  bihi0,bihi1,bihi2,
     &  dx,
     &  beta,
     &  b0,b1,b2,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2,
     &  bilo0,bilo1,bilo2,bihi0,bihi1,bihi2
      REAL
     &  dx(0:NDIM-1),beta,
     &  b0(FACE3d0(bilo,bihi,0)),
     &  b1(FACE3d1(bilo,bihi,0)),
     &  b2(FACE3d2(bilo,bihi,0))
      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))

      integer i,j,k
      REAL d0, d1, d2
c
c***********************************************************************
c
c using face data for b0,b1,b2 will slow down the creation
c of the stencils significantly. we should move to using
c side data for b0,b1,b2

      d0 = -beta / (dx(0)*dx(0))
      d1 = -beta / (dx(1)*dx(1))
      d2 = -beta / (dx(2)*dx(2))

      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               stencil(WW, i, j, k) = d0*b0(i  ,j, k)
               stencil(EE, i, j, k) = d0*b0(i+1,j, k)
               stencil(SS, i, j, k) = d1*b1(j  ,k, i)
               stencil(NN, i, j, k) = d1*b1(j+1,k, i)
               stencil(BB, i, j, k) = d2*b2(k  ,i, j)
               stencil(TT, i, j, k) = d2*b2(k+1,i, j)
            enddo
         enddo
      enddo

      return
      end
c
      recursive subroutine cellpoissonoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2
      REAL dx(0:NDIM-1),beta

      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))

      integer i,j,k
      REAL d0, d1, d2
c
c***********************************************************************
c
      d0 = -beta / (dx(0)*dx(0))
      d1 = -beta / (dx(1)*dx(1))
      d2 = -beta / (dx(2)*dx(2))

      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               stencil(WW, i, j, k) = d0
               stencil(EE, i, j, k) = d0
               stencil(SS, i, j, k) = d1
               stencil(NN, i, j, k) = d1
               stencil(BB, i, j, k) = d2
               stencil(TT, i, j, k) = d2
            enddo
         enddo
      enddo

      return
      end
c
c
      recursive subroutine celldiffusionv1diag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  ailo0,ailo1,ailo2,
     &  aihi0,aihi1,aihi2,
     &  alpha,
     &  a,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer ailo0,ailo1,ailo2
      integer aihi0,aihi1,aihi2
      REAL alpha
      REAL a(CELL3d(ailo,aihi,0))
      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))
      integer i,j,k
c
c***********************************************************************
c

      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               stencil(PP, i,j,k) = -(stencil(WW, i,j,k)
     &                              + stencil(EE, i,j,k)
     &                              + stencil(SS, i,j,k)
     &                              + stencil(NN, i,j,k)
     &                              + stencil(BB, i,j,k)
     &                              + stencil(TT, i,j,k) )
     &                              + alpha * a(i,j,k)
            enddo
         enddo
      enddo

      return
      end
c
c
      recursive subroutine celldiffusionv2diag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  alpha,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2
      REAL alpha
      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))
      integer i,j,k

c
c***********************************************************************
c
      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               stencil(PP, i,j,k) = -(stencil(WW, i,j,k)
     &                              + stencil(EE, i,j,k)
     &                              + stencil(SS, i,j,k)
     &                              + stencil(NN, i,j,k)
     &                              + stencil(BB, i,j,k)
     &                              + stencil(TT, i,j,k) )
     &                              + alpha
            enddo
         enddo
      enddo
c
      return
      end
c
c
      recursive subroutine cellpoissondiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k

c
c***********************************************************************
c

      do k = ifirst2, ilast2
         do j = ifirst1, ilast1
            do i = ifirst0, ilast0
               stencil(PP, i,j,k) = -(stencil(WW, i,j,k)
     &                              + stencil(EE, i,j,k)
     &                              + stencil(SS, i,j,k)
     &                              + stencil(NN, i,j,k)
     &                              + stencil(BB, i,j,k)
     &                              + stencil(TT, i,j,k) )
            enddo
         enddo
      enddo
c
      return
      end
c
c
      recursive subroutine adjcelldiffusiondiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  beta,
     &  b0, b1, b2,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer pfirst0,pfirst1,pfirst2
      integer plast0,plast1,plast2
      integer direction, side, bdrytype
      integer extrapOrder
      REAL dx(0:NDIM-1), beta
      REAL
     &  b0(FACE3d0(ifirst,ilast,0)),
     &  b1(FACE3d1(ifirst,ilast,0)),
     &  b2(FACE3d2(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k
      integer ie0,ie1,ie2

      REAL factor
      REAL b, h
c
c***********************************************************************
c
      h=dx(direction)
      if (bdrytype.eq.ROBIN) then
        b=0.25
        if (direction.eq.0) then
          ie0 = pfirst0-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=beta*b0(ie0,j, k)
                  factor=factor/h*(3.0*h+16.0*b0(ie0,j,k))
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=beta*b0(ie0+1,j,k)
                  factor=factor/h*(3.0*h+16.0*b0(ie0+1,j,k))
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.1) then
          ie1 = pfirst1-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=beta*b1(ie1,k,i)
                  factor=factor/h*(3.0*h+16.0*b1(ie1,k,i))
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=beta*b1(ie1+1,k,i)
                  factor=factor/h*(3.0*h+16.0*b1(ie1+1,k,i))
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.2) then
          ie2 = pfirst2-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=beta*b2(ie2,i,j)
                  factor=factor/h*(3.0*h+16.0*b2(ie2,i,j))
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                enddo
              enddo
            else if (extrapOrder.eq.2) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=beta*b2(ie2+1,i,j)
                  factor=factor/h*(3.0*h+16.0*b2(ie2+1,i,j))
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
                enddo
              enddo
            endif
          endif
        endif
      elseif (bdrytype.eq.NEUMANN) then
      elseif (bdrytype.eq.DIRICHLET) then
        if (direction.eq.0) then
          ie0 = pfirst0-2*side+1
          if(extrapOrder.eq.1) then
            do k = pfirst2, plast2
              do j = pfirst1, plast1
              enddo
           enddo
          else if (extrapOrder.eq.2) then
            do k = pfirst2, plast2
              do j = pfirst1, plast1
                factor = beta*b0(ie0,j,k)/(3.0d0*(h**2))
                stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)+factor
              enddo
            enddo
          endif
        elseif (direction.eq.1) then
          ie1  = pfirst1-2*side+1
          if(extrapOrder.eq.1) then
            do k = pfirst2, plast2
              do i = pfirst0, plast0
              enddo
            enddo
          else if (extrapOrder.eq.2) then
            do k = pfirst2, plast2
              do i = pfirst0, plast0
                factor = beta*b1(ie1,k,i)/(3.0d0*(h**2))
                stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)+factor
              enddo
            enddo
          endif
        elseif (direction.eq.2) then
          ie2  = pfirst2-2*side+1
          if(extrapOrder.eq.1) then
            do j = pfirst1, plast1
              do i = pfirst0, plast0
              enddo
            enddo
          else if (extrapOrder.eq.2) then
            do j = pfirst1, plast1
              do i = pfirst0, plast0
                factor = beta*b2(ie2,i,j)/(3.0d0*(h**2))
                stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)+factor
              enddo
            enddo
          endif
        endif
      endif

      return
      end

      recursive subroutine adjcellpoissondiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,plast0,plast1,plast2,
     &  direction, side, bdrytype
      integer extrapOrder
      REAL dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2

      REAL factor
      REAL b, h
c
c***********************************************************************
c
      h=dx(direction)
      if (bdrytype.eq.ROBIN) then
        b=0.25
        if (direction.eq.0) then
          ie0 = pfirst0-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.1) then
          ie1 = pfirst1-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.2) then
          ie2 = pfirst2-2*side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                enddo
              enddo
            endif
            if (extrapOrder.eq.2) then
              factor=beta/h*(3.0*h+16.0)
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
                enddo
              enddo
            endif
          endif
        endif
      elseif (bdrytype.eq.NEUMANN) then
      elseif (bdrytype.eq.DIRICHLET) then
        if (direction.eq.0) then
          ie0 = pfirst0-2*side+1
          if(extrapOrder.eq.1) then
            do k = pfirst2, plast2
              do j = pfirst1, plast1
              enddo
            enddo
          else if (extrapOrder.eq.2) then
            factor = beta/(3.0d0*(h**2))
            do k = pfirst2, plast2
              do j = pfirst1, plast1
                stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)+factor
              enddo
            enddo
          endif
        elseif (direction.eq.1) then
          ie1  = pfirst1-2*side+1
          if(extrapOrder.eq.1) then
            do k = pfirst2, plast2
              do i = pfirst0, plast0
              enddo
            enddo
          else if (extrapOrder.eq.2) then
            factor = beta/(3.0d0*(h**2))
            do k = pfirst2, plast2
              do i = pfirst0, plast0
                stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)+factor
              enddo
            enddo
          endif
        elseif (direction.eq.2) then
          ie2  = pfirst2-2*side+1
          if(extrapOrder.eq.1) then
            do j = pfirst1, plast1
              do i = pfirst0, plast0
              enddo
            enddo
          else if (extrapOrder.eq.2) then
            factor = beta/(3.0d0*(h**2))
            do j = pfirst1, plast1
              do i = pfirst0, plast0
                stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)+factor
              enddo
            enddo
          endif
        endif
      endif
      return
      end

      recursive subroutine adjcelldiffusioncfdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  r,
     &  direction, side,
     &  interporder,
     &  dx,
     &  beta,
     &  b0, b1, b2,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,plast0,plast1,plast2,
     &  direction, side,
     &  interporder
      integer r
      REAL
     &  dx(0:NDIM-1), beta
      REAL
     &  b0(FACE3d0(ifirst,ilast,0)),
     &  b1(FACE3d1(ifirst,ilast,0)),
     &  b2(FACE3d2(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2,edge

      REAL factor
      REAL h

      REAL dr,dr1

c r is the refinement ratio, convert to double precision
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      h=dx(direction)
      if (direction.eq.0) then
         ie0 = pfirst0-2*side+1
         edge = pfirst0-side+1
         if (interporder .eq. 1) then
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  factor=beta*b0(edge,j,k)/(3.0*(h**2))
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  factor=dr1*beta*b0(edge,j,k)/(h**2)
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
               enddo
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-2*side+1
         edge = pfirst1-side+1
         if (interporder .eq. 1) then
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  factor=beta*b1(edge,k,i)/(3.0*(h**2))
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  factor=dr1*beta*b1(edge,k,i)/(h**2)
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
               enddo
            enddo
         endif
      elseif (direction.eq.2) then
         ie2 = pfirst2-2*side+1
         edge = pfirst2-side+1
         if (interporder .eq. 1) then
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  factor=beta*b2(edge,i,j)/(3.0*(h**2))
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  factor=dr1*beta*b2(edge,i,j)/(h**2)
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
               enddo
            enddo
         endif
      endif

      return
      end
c
      recursive subroutine adjcellpoissoncfdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  r,
     &  direction, side,
     &  interporder,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,plast0,plast1,plast2,
     &  direction, side,
     &  interporder
      integer r
      REAL dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2

      REAL factor
      REAL h
      REAL dr,dr1
c r is the refinement ratio, convert to double precision
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      h=dx(direction)
      if (direction.eq.0) then
         ie0 = pfirst0-2*side+1
         if (interporder .eq. 1) then
            factor=beta/(3.0*(h**2))
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            factor=dr1*beta/(h**2)
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  stencil(PP,ie0,j,k)=stencil(PP,ie0,j,k)-factor
               enddo
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-2*side+1
         if (interporder .eq. 1) then
            factor=beta/(3.0*(h**2))
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            factor=dr1*beta/(h**2)
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  stencil(PP,i,ie1,k)=stencil(PP,i,ie1,k)-factor
               enddo
            enddo
         endif
      elseif (direction.eq.2) then
         ie2 = pfirst2-2*side+1
         if (interporder .eq. 1) then
            factor=beta/(3.0*(h**2))
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
               enddo
            enddo
         else if (interporder .eq. 2) then
            factor=dr1*beta/(h**2)
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  stencil(PP,i,j,ie2)=stencil(PP,i,j,ie2)-factor
               enddo
            enddo
         endif
      endif

      return
      end
c
c
      recursive subroutine adjcelldiffusionoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  dirfactor, neufactor,
     &  beta,
     &  b0, b1, b2,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer pfirst0,pfirst1,pfirst2
      integer plast0,plast1,plast2
      integer direction, side
      integer bdrytype, extrapOrder
      REAL dirfactor, neufactor
      REAL dx(0:NDIM-1), beta
      REAL
     &  b0(FACE3d0(ifirst,ilast,0)),
     &  b1(FACE3d1(ifirst,ilast,0)),
     &  b2(FACE3d2(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2
      REAL factor
      REAL h
c
c***********************************************************************
c

      h=dx(direction)

      if (bdrytype.eq.ROBIN) then
        if (direction.eq.0) then
          ie0 = pfirst0-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=(2.0*h)/(4.0*b0(ie0,j,k)+h)
                  stencil(WW,ie0,j,k)=factor*stencil(WW,ie0,j,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=beta*b0(ie0,j,k)
                  factor=factor/(h*(3.0*h+16.0*b0(ie0,j,k)))
                  stencil(WW,ie0,j,k)=-9.0*factor
                  stencil(EE,ie0,j,k)=stencil(EE,ie0,j,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=2.0*h/(4.0*b0(ie0,j,k)+h)
                  stencil(EE,ie0-1,j,k)=factor*stencil(EE,ie0-1,j,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  factor=beta*b0(ie0,j,k)
                  factor=factor/(h*(3.0*h+16.0*b0(ie0,j,k)))
                  stencil(EE,ie0-1,j,k)=-9.0*factor
                  stencil(WW,ie0-1,j,k)=stencil(WW,ie0-1,j,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.1) then
          ie1 = pfirst1-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=(2.0*h)/(4.0*b1(ie1,k,i)+h)
                  stencil(SS,i,ie1,k)=factor*stencil(SS,i,ie1,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=beta*b1(ie1,k,i)
                  factor=factor/(h*(3.0*h+16.0*b1(ie1,k,i)))
                  stencil(SS,i,ie1,k)=-9.0*factor
                  stencil(NN,i,ie1,k)=stencil(NN,i,ie1,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=2.0*h/(4.0*b1(ie1,k,i)+h)
                  stencil(NN,i,ie1-1,k)=factor*stencil(NN,i,ie1-1,k)
              enddo
              enddo
            else if(extrapOrder.eq.2) then
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  factor=beta*b1(ie1,k,i)
                  factor=factor/(h*(3.0*h+16.0*b1(ie1,k,i)))
                  stencil(NN,i,ie1-1,k)=-9.0*factor
                  stencil(SS,i,ie1-1,k)=stencil(SS,i,ie1-1,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.2) then
          ie2 = pfirst2-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=(2.0*h)/(4.0*b2(ie2,i,j)+h)
                  stencil(BB,i,j,ie2)=factor*stencil(BB,i,j,ie2)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=beta*b2(ie2,i,j)
                  factor=factor/(h*(3.0*h+16.0*b2(ie2,i,j)))
                  stencil(BB,i,j,ie2)=-9.0*factor
                  stencil(TT,i,j,ie2)=stencil(TT,i,j,ie2)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=2.0*h/(4.0*b2(ie2,i,j)+h)
                  stencil(TT,i,j,ie2-1)=factor*stencil(TT,i,j,ie2-1)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  factor=beta*b2(ie2,i,j)
                  factor=factor/(h*(3.0*h+16.0*b2(ie2,i,j)))
                  stencil(TT,i,j,ie2-1)=-9.0*factor
                  stencil(BB,i,j,ie2-1)=stencil(BB,i,j,ie2-1)-factor
                enddo
              enddo
            endif
          endif
        endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         if (direction.eq.0) then
           ie0 = pfirst0-side+1
           if( side. eq. 0) then
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(WW,ie0,j,k) = stencil(WW,ie0,j,k)*factor
                 enddo
               enddo
             endif
             if(extrapOrder.eq.2) then
                do k = pfirst2, plast2
                  do j = pfirst1, plast1
                    stencil(EE,ie0,j,k) = stencil(EE,ie0,j,k)
     &                                   +stencil(WW,ie0,j,k)/3.0
                    stencil(WW,ie0,j,k) = 8.0*stencil(WW,ie0,j,k)/3.0
                  enddo
                enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(EE,ie0-1,j,k) = stencil(EE,ie0-1,j,k)*factor
                 enddo
               enddo
             endif
             if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(WW,ie0-1,j,k)=stencil(WW,ie0-1,j,k)
     &                                  +stencil(EE,ie0-1,j,k)/3.0
                   stencil(EE,ie0-1,j,k)= 8.0*stencil(EE,ie0-1,j,k)/3.0
                 enddo
               enddo
             endif
           endif
         elseif (direction.eq.1) then
           ie1 = pfirst1-side+1
           if( side .eq. 0) then
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                    stencil(SS,i,ie1,k) = stencil(SS,i,ie1,k)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                   stencil(NN,i,ie1,k)=stencil(NN,i,ie1,k)
     &                                +stencil(SS,i,ie1,k)/3.0
                   stencil(SS,i,ie1,k)=8.0*stencil(SS,i,ie1,k)/3.0
                 enddo
               enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                    stencil(NN,i,ie1-1,k) = stencil(NN,i,ie1-1,k)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                   stencil(SS,i,ie1-1,k)=stencil(SS,i,ie1-1,k)
     &                                  +stencil(NN,i,ie1-1,k)/3.0
                   stencil(NN,i,ie1-1,k)=8.0*stencil(NN,i,ie1-1,k)/3.0
                 enddo
               enddo
             endif
           endif
         elseif (direction.eq.2) then
           ie2 = pfirst2-side+1
           if( side .eq. 0) then
             if(extrapOrder.eq.1) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                    stencil(BB,i,j,ie2) = stencil(BB,i,j,ie2)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                   stencil(TT,i,j,ie2)=stencil(TT,i,j,ie2)
     &                                +stencil(BB,i,j,ie2)/3.0
                   stencil(BB,i,j,ie2)=8.0*stencil(BB,i,j,ie2)/3.0
                 enddo
               enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                    stencil(TT,i,j,ie2-1) = stencil(TT,i,j,ie2-1)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                   stencil(BB,i,j,ie2-1)=stencil(BB,i,j,ie2-1)
     &                                  +stencil(TT,i,j,ie2-1)/3.0
                   stencil(TT,i,j,ie2-1)=8.0*stencil(TT,i,j,ie2-1)/3.0
                 enddo
               enddo
             endif
           endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if(side .eq. 0) then
               do k = pfirst2, plast2
                  do j = pfirst1, plast1
                     stencil(WW,ie0,j,k) = 0.0d0
                  enddo
               enddo
            else
               do k = pfirst2, plast2
                  do j = pfirst1, plast1
                     stencil(EE,ie0-1,j,k) = 0.0d0
                  enddo
               enddo
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if(side .eq. 0) then
               do k = pfirst2, plast2
                  do i = pfirst0, plast0
                     stencil(SS,i,ie1,k) = 0.0d0
                  enddo
               enddo
            else
               do k = pfirst2, plast2
                  do i = pfirst0, plast0
                     stencil(NN,i,ie1-1,k) = 0.0d0
                  enddo
               enddo
            endif
         elseif (direction.eq.2) then
            ie2 = pfirst2-side+1
            if(side .eq. 0) then
               do j = pfirst1, plast1
                  do i = pfirst0, plast0
                     stencil(BB,i,j,ie2) = 0.0d0
                  enddo
               enddo
            else
               do j = pfirst1, plast1
                  do i = pfirst0, plast0
                     stencil(TT,i,j,ie2-1) = 0.0d0
                  enddo
               enddo
            endif
         endif
      endif
c
      return
      end
c
      recursive subroutine adjcellpoissonoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  dirfactor, neufactor,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ifirst2,ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,plast0,plast1,plast2,
     &  direction, side, bdrytype,
     &  extrapOrder
      REAL dirfactor, neufactor
      REAL dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:6, CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2
      REAL factor
      REAL h
c
c***********************************************************************
c
c for now we will only do the dir=0 robin boundary conditions
c we will also assume homogenous boundary conditions for now
c
      h=dx(direction)

      if (bdrytype.eq.ROBIN) then
        if (direction.eq.0) then
          ie0 = pfirst0-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              factor=(2.0*h)/(4.0+h)
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(WW,ie0,j,k)=factor*stencil(WW,ie0,j,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0))
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(WW,ie0,j,k)=-9.0*factor
                  stencil(EE,ie0,j,k)=stencil(EE,ie0,j,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              factor=2.0*h/(4.0+h)
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(EE,ie0-1,j,k)=factor*stencil(EE,ie0-1,j,k)
              enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0))
              do k = pfirst2, plast2
                do j = pfirst1, plast1
                  stencil(EE,ie0-1,j,k)=-9.0*factor
                  stencil(WW,ie0-1,j,k)=stencil(WW,ie0-1,j,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.1) then
          ie1 = pfirst1-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              factor=(2.0*h)/(4.0+h)
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(SS,i,ie1,k)=factor*stencil(SS,i,ie1,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0))
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(SS,i,ie1,k)=-9.0*factor
                  stencil(NN,i,ie1,k)=stencil(NN,i,ie1,k)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              factor=2.0*h/(4.0+h)
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(NN,i,ie1-1,k)=factor*stencil(NN,i,ie1-1,k)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0d0))
              do k = pfirst2, plast2
                do i = pfirst0, plast0
                  stencil(NN,i,ie1-1,k)=-9.0*factor
                  stencil(SS,i,ie1-1,k)=stencil(SS,i,ie1-1,k)-factor
                enddo
              enddo
            endif
          endif
        elseif (direction.eq.2) then
          ie2 = pfirst2-side+1
          if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
              factor=(2.0*h)/(4.0+h)
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(BB,i,j,ie2)=factor*stencil(BB,i,j,ie2)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0))
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(BB,i,j,ie2)=-9.0*factor
                  stencil(TT,i,j,ie2)=stencil(TT,i,j,ie2)-factor
                enddo
              enddo
            endif
          else
            if(extrapOrder.eq.1) then
              factor=2.0*h/(4.0+h)
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(TT,i,j,ie2-1)=factor*stencil(TT,i,j,ie2-1)
                enddo
              enddo
            else if(extrapOrder.eq.2) then
              factor=beta/(h*(3.0*h+16.0))
              do j = pfirst1, plast1
                do i = pfirst0, plast0
                  stencil(TT,i,j,ie2-1)=-9.0*factor
                  stencil(BB,i,j,ie2-1)=stencil(BB,i,j,ie2-1)-factor
                enddo
              enddo
            endif
          endif
        endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         if (direction.eq.0) then
           ie0 = pfirst0-side+1
           if( side. eq. 0) then
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(WW,ie0,j,k) = stencil(WW,ie0,j,k)*factor
                 enddo
               enddo
             elseif(extrapOrder.eq.2) then
                do k = pfirst2, plast2
                  do j = pfirst1, plast1
                    stencil(EE,ie0,j,k) = stencil(EE,ie0,j,k)
     &                                   +stencil(WW,ie0,j,k)/3.0
                    stencil(WW,ie0,j,k) = 8.0*stencil(WW,ie0,j,k)/3.0
                  enddo
                enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(EE,ie0-1,j,k) = stencil(EE,ie0-1,j,k)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do j = pfirst1, plast1
                   stencil(WW,ie0-1,j,k)=stencil(WW,ie0-1,j,k)
     &                                  +stencil(EE,ie0-1,j,k)/3.0
                   stencil(EE,ie0-1,j,k)= 8.0*stencil(EE,ie0-1,j,k)/3.0
                 enddo
               enddo
             endif
           endif
         elseif (direction.eq.1) then
           ie1 = pfirst1-side+1
           if( side .eq. 0) then
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                    stencil(SS,i,ie1,k) = stencil(SS,i,ie1,k)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                   stencil(NN,i,ie1,k)=stencil(NN,i,ie1,k)
     &                                +stencil(SS,i,ie1,k)/3.0
                   stencil(SS,i,ie1,k)=8.0*stencil(SS,i,ie1,k)/3.0
                 enddo
               enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                    stencil(NN,i,ie1-1,k) = stencil(NN,i,ie1-1,k)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do k = pfirst2, plast2
                 do i = pfirst0, plast0
                   stencil(SS,i,ie1-1,k)=stencil(SS,i,ie1-1,k)
     &                                  +stencil(NN,i,ie1-1,k)/3.0
                   stencil(NN,i,ie1-1,k)=8.0*stencil(NN,i,ie1-1,k)/3.0
                 enddo
               enddo
             endif
           endif
         elseif (direction.eq.2) then
           ie2 = pfirst2-side+1
           if( side .eq. 0) then
             if(extrapOrder.eq.1) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                    stencil(BB,i,j,ie2) = stencil(BB,i,j,ie2)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                   stencil(TT,i,j,ie2)=stencil(TT,i,j,ie2)
     &                                +stencil(BB,i,j,ie2)/3.0
                   stencil(BB,i,j,ie2)=8.0*stencil(BB,i,j,ie2)/3.0
                 enddo
               enddo
             endif
           else
             if(extrapOrder.eq.1) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                    stencil(TT,i,j,ie2-1) = stencil(TT,i,j,ie2-1)*factor
                 enddo
               enddo
             else if(extrapOrder.eq.2) then
               do j = pfirst1, plast1
                 do i = pfirst0, plast0
                   stencil(BB,i,j,ie2-1)=stencil(BB,i,j,ie2-1)
     &                                  +stencil(TT,i,j,ie2-1)/3.0
                   stencil(TT,i,j,ie2-1)=8.0*stencil(TT,i,j,ie2-1)/3.0
                 enddo
               enddo
             endif
           endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if(side .eq. 0) then
               do k = pfirst2, plast2
                  do j = pfirst1, plast1
                     stencil(WW,ie0,j,k) = 0.0d0
                  enddo
               enddo
            else
               do k = pfirst2, plast2
                  do j = pfirst1, plast1
                     stencil(EE,ie0-1,j,k) = 0.0d0
                  enddo
               enddo
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if(side .eq. 0) then
               do k = pfirst2, plast2
                  do i = pfirst0, plast0
                     stencil(SS,i,ie1,k) = 0.0d0
                  enddo
               enddo
            else
               do k = pfirst2, plast2
                  do i = pfirst0, plast0
                     stencil(NN,i,ie1-1,k) = 0.0d0
                  enddo
               enddo
            endif
         elseif (direction.eq.2) then
            ie2 = pfirst2-side+1
            if(side .eq. 0) then
               do j = pfirst1, plast1
                  do i = pfirst0, plast0
                     stencil(BB,i,j,ie2) = 0.0d0
                  enddo
               enddo
            else
               do j = pfirst1, plast1
                  do i = pfirst0, plast0
                     stencil(TT,i,j,ie2-1) = 0.0d0
                  enddo
               enddo
            endif
         endif
      endif

      return
      end
c
c
      recursive subroutine adjcelldiffusioncfoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  cfirst0,cfirst1,cfirst2,
     & clast0,clast1,clast2,
     &  r,
     &  direction, side,
     &  interporder,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer cfirst0,cfirst1,cfirst2
      integer clast0,clast1,clast2
      integer direction, side
      integer r
      integer interporder
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2
      integer offset
      REAL dr

c r is the refinement ratio, convert to double precision
      dr = dfloat(r)
c
c***********************************************************************
      offset=1-2*side
      if (direction.eq.0) then
         ie0 = cfirst0-side+1
         if(side.eq.0) then
            if(interporder .eq. 1) then
               do k = cfirst2, clast2
                  do j = cfirst1, clast1
                     stencil(WW,ie0,j,k)=2.0*stencil(WW,ie0,j,k)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do k = cfirst2, clast2
                  do j = cfirst1, clast1
                     stencil(EE,ie0,j,k) = stencil(EE,ie0,j,k)
     &                           -stencil(WW,ie0,j,k)*(dr-1.0)/(dr+3.0)
                     stencil(WW,ie0,j,k)=
     &                       8.0*stencil(WW,ie0,j,k)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         else
            if(interporder .eq. 1) then
               do k = cfirst2, clast2
                  do j = cfirst1, clast1
                     stencil(EE,ie0-1,j,k)=2.0*stencil(EE,ie0-1,j,k)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do k = cfirst2, clast2
                  do j = cfirst1, clast1
                     stencil(WW,ie0-1,j,k) = stencil(WW,ie0-1,j,k)
     &                    -stencil(EE,ie0-1,j,k)*(dr-1.0)/(dr+3.0)
                     stencil(EE,ie0-1,j,k)=
     &                    8.0*stencil(EE,ie0-1,j,k)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         endif
      elseif (direction.eq.1) then
         ie1 = cfirst1-side+1
         if(side.eq.0) then
            if(interporder .eq. 1) then
               do k = cfirst2, clast2
                  do i = cfirst0, clast0
                     stencil(SS,i,ie1,k)=2.0*stencil(SS,i,ie1,k)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do k = cfirst2, clast2
                  do i = cfirst0, clast0
                     stencil(NN,i,ie1,k) = stencil(NN,i,ie1,k)
     &                    -stencil(SS,i,ie1,k)*(dr-1.0)/(dr+3.0)
                     stencil(SS,i,ie1,k)=
     &                    8.0*stencil(SS,i,ie1,k)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         else
            if(interporder .eq. 1) then
               do k = cfirst2, clast2
                  do i = cfirst0, clast0
                     stencil(NN,i,ie1-1,k)=2.0*stencil(NN,i,ie1-1,k)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do k = cfirst2, clast2
                  do i = cfirst0, clast0
                     stencil(SS,i,ie1-1,k) = stencil(SS,i,ie1-1,k)
     &                    -stencil(NN,i,ie1-1,k)*(dr-1.0)/(dr+3.0)
                     stencil(NN,i,ie1-1,k)=
     &                    8.0*stencil(NN,i,ie1-1,k)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         endif
      elseif (direction.eq.2) then
         ie2 = cfirst2-side+1
         if(side.eq.0) then
            if(interporder .eq. 1) then
               do j = cfirst1, clast1
                  do i = cfirst0, clast0
                     stencil(BB,i,j,ie2)=2.0*stencil(BB,i,j,ie2)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do j = cfirst1, clast1
                  do i = cfirst0, clast0
                     stencil(TT,i,j,ie2) = stencil(TT,i,j,ie2)
     &                    -stencil(BB,i,j,ie2)*(dr-1.0)/(dr+3.0)
                     stencil(BB,i,j,ie2)=
     &                    8.0*stencil(BB,i,j,ie2)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         else
            if(interporder .eq. 1) then
               do j = cfirst1, clast1
                  do i = cfirst0, clast0
                     stencil(TT,i,j,ie2-1)=2.0*stencil(TT,i,j,ie2-1)/3.0
                  enddo
               enddo
            else if(interporder .eq. 2) then
               do j = cfirst1, clast1
                  do i = cfirst0, clast0
                     stencil(BB,i,j,ie2-1) = stencil(BB,i,j,ie2-1)
     &                    -stencil(TT,i,j,ie2-1)*(dr-1.0)/(dr+3.0)
                     stencil(TT,i,j,ie2-1)=
     &                    8.0*stencil(TT,i,j,ie2-1)/((dr+1.0)*(dr+3.0))
                  enddo
               enddo
            endif
         endif
      endif
      return
      end
c
c
      recursive subroutine readjcelldiffusionoffdiag3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ifirst2
      integer ilast0,ilast1,ilast2
      integer pfirst0,pfirst1,pfirst2
      integer plast0,plast1,plast2
      integer direction, side
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer i,j,k,ie0,ie1,ie2
c
c***********************************************************************
c

      if (direction.eq.0) then
         ie0 = pfirst0-side+1
         if(side .eq. 0) then
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  stencil(WW,ie0,j,k) = 0.0
               enddo
            enddo
         else
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  stencil(EE,ie0-1,j,k) = 0.0
               enddo
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-side+1
         if(side .eq. 0) then
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  stencil(SS,i,ie1,k) = 0.0
               enddo
            enddo
         else
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  stencil(NN,i,ie1-1,k) = 0.0
               enddo
            enddo
         endif
      elseif (direction.eq.2) then
         ie2 = pfirst2-side+1
         if(side .eq. 0) then
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  stencil(BB,i,j,ie2) = 0.0
               enddo
            enddo
         else
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  stencil(TT,i,j,ie2-1) = 0.0
               enddo
            enddo
         endif
      endif
c
      return
      end
c
c
      recursive subroutine adjcelldiffusioncfbdryrhs3d(
     &  ifirst0,ifirst1,ifirst2,
     &  ilast0,ilast1,ilast2,
     &  pfirst0,pfirst1,pfirst2,
     &  plast0,plast1,plast2,
     &  direction, side,
     &  sgcw,
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
      integer sgcw
      REAL stencil(0:6,CELL3d(ifirst,ilast,sgcw))
      integer gcw
      REAL u(CELL3d(ifirst,ilast,gcw))
      REAL rhs(CELL3d(ifirst,ilast,0))
      integer i,j,k,ie0,ie1,ie2
c
c***********************************************************************
c

      if (direction.eq.0) then
         ie0 = pfirst0+1-(2*side)
         if(side.eq.0) then
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  rhs(ie0,j,k) = rhs(ie0,j,k) -
     &                 stencil(WW,ie0,j,k)*u(ie0-1,j,k)
               enddo
            enddo
         else
            do k = pfirst2, plast2
               do j = pfirst1, plast1
                  rhs(ie0,j,k) = rhs(ie0,j,k) -
     &                 stencil(EE,ie0,j,k)*u(ie0+1,j,k)
               enddo
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1+1-(2*side)
         if(side.eq.0) then
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  rhs(i,ie1,k) = rhs(i,ie1,k)-
     &                 stencil(SS,i,ie1,k)*u(i,ie1-1,k)
               enddo
            enddo
         else
            do k = pfirst2, plast2
               do i = pfirst0, plast0
                  rhs(i,ie1,k) = rhs(i,ie1,k)-
     &                 stencil(NN,i,ie1,k)*u(i,ie1+1,k)
               enddo
            enddo
         endif
      elseif (direction.eq.2) then
         ie2 = pfirst2+1-(2*side)
         if(side.eq.0) then
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  rhs(i,j,ie2) = rhs(i,j,ie2)-
     &                 stencil(BB,i,j,ie2)*u(i,j,ie2-1)
               enddo
            enddo
         else
            do j = pfirst1, plast1
               do i = pfirst0, plast0
                  rhs(i,j,ie2) = rhs(i,j,ie2)-
     &                 stencil(TT,i,j,ie2)*u(i,j,ie2+1)
               enddo
            enddo
         endif
      endif
      return
      end
