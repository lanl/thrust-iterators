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
c  Revision:    $Revision: 2727 $
c  Modified:    $Date: 2006-06-22 15:52:36 -0600 (Thu, 22 Jun 2006) $
c  Description: F77 routines that compute matrix entries for 2d cell centered diffusion solver.
c
define(NDIM,2)dnl
define(REAL,`double precision')dnl
define(PP,0)dnl
define(WW,1)dnl
define(EE,2)dnl
define(SS,3)dnl
define(NN,4)dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim2d.i)dnl
c
c
      recursive subroutine celldiffusionoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  bilo0,bilo1,bihi0,bihi1,
     &  dx,
     &  beta,
     &  b0,b1,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  bilo0,bilo1,bihi0,bihi1
      REAL
     &  dx(0:NDIM-1),beta,
     &  b0(FACE2d0(bilo,bihi,0)),
     &  b1(FACE2d1(bilo,bihi,0))
      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))

      integer ie0,ic1
      REAL d0, d1
c
c***********************************************************************
c
c using face data for b0,b1 will slow down the creation
c of the stencils significantly. we should move to using
c side data for b0,b1
      
      d0 = -beta / (dx(0)*dx(0))
      d1 = -beta / (dx(1)*dx(1))
      
      do ic1 = ifirst1, ilast1
         do ie0 = ifirst0, ilast0
            stencil(WW, ie0  ,ic1) = d0*b0(ie0  ,ic1) 
            stencil(EE, ie0  ,ic1) = d0*b0(ie0+1,ic1) 
            stencil(SS, ie0  ,ic1) = d1*b1(ic1  ,ie0 ) 
            stencil(NN, ie0  ,ic1) = d1*b1(ic1+1,ie0) 
         enddo
      enddo
      
      return
      end
c     
      recursive subroutine cellpoissonoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1
      REAL dx(0:NDIM-1),beta

      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))

      integer ie0,ic1
      REAL d0, d1
c
c***********************************************************************
c
      d0 = -beta / (dx(0)*dx(0))
      d1 = -beta / (dx(1)*dx(1))
      
      do ic1 = ifirst1, ilast1
         do ie0 = ifirst0, ilast0
            stencil(WW, ie0  ,ic1) = d0
            stencil(EE, ie0  ,ic1) = d0
            stencil(SS, ie0  ,ic1) = d1
            stencil(NN, ie0  ,ic1) = d1
         enddo
      enddo
      
      return
      end
c
c
      recursive subroutine celldiffusionv1diag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  ailo0,ailo1,aihi0,aihi1,
     &  alpha,
     &  a,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  ailo0,ailo1,aihi0,aihi1
      REAL
     &  alpha,
     &  a(CELL2d(ailo,aihi,0))
      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1
c
c***********************************************************************
c

      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            stencil(PP, ic0,ic1) = -(stencil(WW, ic0,ic1) 
     &                             + stencil(EE, ic0,ic1) 
     &                             + stencil(SS, ic0,ic1) 
     &                             + stencil(NN, ic0,ic1) )
     &                      + alpha * a(ic0,ic1)
         enddo
      enddo
c
      return
      end
c
c     
      recursive subroutine celldiffusionv2diag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  alpha,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ilast0,ilast1
      REAL alpha
      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1

c
c***********************************************************************
c
      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            stencil(PP, ic0,ic1) = -(stencil(WW, ic0,ic1) 
     &                             + stencil(EE, ic0,ic1) 
     &                             + stencil(SS, ic0,ic1) 
     &                             + stencil(NN, ic0,ic1) )
     &                             + alpha
         enddo
      enddo
c
      return
      end
c
c     
      recursive subroutine cellpoissondiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ifirst1,ilast0,ilast1
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1

c     
c***********************************************************************
c

      do ic1 = ifirst1, ilast1
         do ic0 = ifirst0, ilast0
            stencil(PP, ic0,ic1) = -(stencil(WW, ic0,ic1) 
     &                             + stencil(EE, ic0,ic1) 
     &                             + stencil(SS, ic0,ic1) 
     &                             + stencil(NN, ic0,ic1) )
         enddo
      enddo
c
      return
      end
c
c
      recursive subroutine adjcelldiffusiondiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  beta,
     &  b0, b1,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side, bdrytype
      integer extrapOrder
      REAL 
     &  dx(0:NDIM-1), beta
      REAL
     &  b0(FACE2d0(ifirst,ilast,0)),
     &  b1(FACE2d1(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1, edge
      
      REAL factor
      REAL b, h
c
c***********************************************************************
c
      if (bdrytype.eq.ROBIN) then
         if (direction.eq.0) then
            ie0 = pfirst0-2*side+1
            edge= pfirst0-side+1
            b=0.25
            h=dx(0)
            if (side .eq. 0 ) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                  else if (extrapOrder.eq.2) then
                    factor=beta*b0(edge,ic1)/h*(3.0*h+16.0*b0(edge,ic1))
                    stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
                  endif
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                  else if (extrapOrder.eq.2) then
                    factor=beta*b0(edge,ic1)/h*(3.0*h+16.0*b0(edge,ic1))
                    stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
                  endif
               end do
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-2*side+1
            do ic0 = pfirst0, plast0
            enddo
         endif

      else
         if (bdrytype.eq.NEUMANN) then
            if (direction.eq.0) then
               ie0 = pfirst0-side+1
               do ic1 = pfirst1, plast1
               enddo
            elseif (direction.eq.1) then
               ie1  = pfirst1-side+1
               do ic0 = pfirst0, plast0
               enddo
            endif
         elseif (bdrytype.eq.DIRICHLET) then
            if (direction.eq.0) then
               ie0 = pfirst0-side+1
               h=dx(0)
               if(extrapOrder.eq.1) then
                  do ic1 = pfirst1, plast1
                  enddo
               else if (extrapOrder.eq.2) then
                  do ic1 = pfirst1, plast1
                     factor = beta*b0(ie0,ic1)/(3.0d0*(h**2))
                     stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)+factor
                  enddo
               endif
            elseif (direction.eq.1) then
               ie1  = pfirst1-side+1
               h=dx(1)
               if(extrapOrder.eq.1) then
                  do ic0 = pfirst0, plast0
                  enddo
               else if (extrapOrder.eq.2) then
                  do ic0 = pfirst0, plast0
                     factor = beta*b1(ie1,ic0)/(3.0d0*(h**2))
                     stencil(PP,ic0,ie1)=stencil(PP,ic0,ie1)+factor
                  enddo
               endif
            endif
         endif         
      endif
      
      return
      end

      recursive subroutine adjcellpoissondiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
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
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side, bdrytype
      integer extrapOrder
      REAL 
     &  dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1, edge
      
      REAL factor
      REAL b, h
c
c***********************************************************************
c
      if (bdrytype.eq.ROBIN) then
         if (direction.eq.0) then
            ie0 = pfirst0-2*side+1
            edge= pfirst0-side+1
            b=0.25
            h=dx(0)
            if (side .eq. 0 ) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                  else if (extrapOrder.eq.2) then
                    factor=beta/h*(3.0*h+16.0)
                    stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
                  endif
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                  else if (extrapOrder.eq.2) then
                    factor=beta/h*(3.0*h+16.0)
                    stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
                  endif
               end do
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-2*side+1
            do ic0 = pfirst0, plast0
            enddo
         endif

      else
         if (bdrytype.eq.NEUMANN) then
            if (direction.eq.0) then
               ie0 = pfirst0-side+1
               do ic1 = pfirst1, plast1
               enddo
            elseif (direction.eq.1) then
               ie1  = pfirst1-side+1
               do ic0 = pfirst0, plast0
               enddo
            endif
         elseif (bdrytype.eq.DIRICHLET) then
            if (direction.eq.0) then
               ie0 = pfirst0-side+1
               do ic1 = pfirst1, plast1
               enddo
            elseif (direction.eq.1) then
               ie1  = pfirst1-side+1
               do ic0 = pfirst0, plast0
               enddo
            endif
         endif         
      endif

      return
      end

      recursive subroutine adjcelldiffusioncfdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  r,
     &  direction, side,
     &  interporder,
     &  dx,
     &  beta,
     &  b0, b1,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  interporder
      integer r
      REAL 
     &  dx(0:NDIM-1), beta
      REAL
     &  b0(FACE2d0(ifirst,ilast,0)),
     &  b1(FACE2d1(ifirst,ilast,0))
      integer sgcw
      REAL
     &  stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1, edge
      
      REAL factor
      REAL h

      REAL dr,dr1

c r is the refinement ratio, convert to double precision      
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      if (direction.eq.0) then
         ie0 = pfirst0-2*side+1
         edge= pfirst0-side+1
         h=dx(0)
         if (interporder .eq. 1) then
            do ic1 = pfirst1, plast1
               factor=beta*b0(edge,ic1)/(3.0*(h**2))
               stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
            enddo
         else if (interporder .eq. 2) then
            do ic1 = pfirst1, plast1
             factor=dr1*beta*b0(edge,ic1)/(h**2)
             stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-2*side+1
         edge= pfirst1-side+1
         h=dx(1)
         if (interporder .eq. 1) then
            do ic0 = pfirst0, plast0
               factor=beta*b1(edge,ic0)/(3.0*(h**2))
               stencil(PP,ic0,ie1)=stencil(PP,ic0,ie1)-factor
            enddo
         else if (interporder .eq. 2) then
            do ic0 = pfirst0, plast0
             factor=dr1*beta*b1(edge,ic0)/(h**2)
             stencil(PP,ic0,ie1)=stencil(PP,ic0,ie1)-factor
            enddo
         endif
      endif
      
      return
      end
c
      recursive subroutine adjcellpoissoncfdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
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
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  interporder
      integer r
      REAL 
     &  dx(0:NDIM-1), beta
      integer sgcw
      REAL
     &  stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1, edge
      
      REAL factor
      REAL h
      REAL dr,dr1
c r is the refinement ratio, convert to double precision      
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      if (direction.eq.0) then
         ie0 = pfirst0-2*side+1
         edge= pfirst0-side+1
         h=dx(0)
         if (interporder .eq. 1) then
            do ic1 = pfirst1, plast1
               factor=beta/(3.0*(h**2))
               stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
            enddo
         else if (interporder .eq. 2) then
            do ic1 = pfirst1, plast1
               factor=dr1*beta/(h**2)
               stencil(PP,ie0,ic1)=stencil(PP,ie0,ic1)-factor
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-2*side+1
         edge= pfirst1-side+1
         h=dx(1)
         if (interporder .eq. 1) then
            do ic0 = pfirst0, plast0
               factor=beta/(3.0*(h**2))
               stencil(PP,ic0,ie1)=stencil(PP,ic0,ie1)-factor
            enddo
         else if (interporder .eq. 2) then
            do ic0 = pfirst0, plast0
               factor=dr1*beta/(h**2)
               stencil(PP,ic0,ie1)=stencil(PP,ic0,ie1)-factor
            enddo
         endif
      endif
      
      return
      end
c
c
      recursive subroutine adjcelldiffusionoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  dirfactor, neufactor, 
     &  beta,
     &  b0, b1,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side, bdrytype,
     &  extrapOrder
      REAL
     &  dirfactor, neufactor
      REAL 
     &  dx(0:NDIM-1), beta
      REAL
     &  b0(FACE2d0(ifirst,ilast,0)),
     &  b1(FACE2d1(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1
      REAL factor
      REAL h
c
c***********************************************************************
c

      if (bdrytype.eq.ROBIN) then
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            h=dx(0)
            if (side .eq. 0 ) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                    factor=(2.0*h)/(4.0*b0(ie0,ic1)+h)
                    stencil(WW,ie0,ic1)=factor*stencil(WW,ie0,ic1)
                  else if(extrapOrder.eq.2) then
                    factor=beta*b0(ie0,ic1)/(h*(3.0*h+16.0*b0(ie0,ic1)))
                    stencil(WW,ie0,ic1)=-9.0*factor
                    stencil(EE,ie0,ic1)=stencil(EE,ie0,ic1)-factor 
                  endif                 
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                    factor=2.0*h/(4.0*b0(ie0,ic1)+h)
                    stencil(EE,ie0-1,ic1)=factor*stencil(EE,ie0-1,ic1)
                  else if(extrapOrder.eq.2) then
                    factor=beta*b0(ie0,ic1)/(h*(3.0*h+16.0*b0(ie0,ic1)))
                    stencil(EE,ie0-1,ic1)=-9.0*factor
                    stencil(WW,ie0-1,ic1)=stencil(WW,ie0-1,ic1)-factor
                  endif
               enddo
            endif
         elseif (direction.eq.1) then
c Robin bc's for dir=1 not implemented
            ie1 = pfirst1-side+1
            do ic0 = pfirst0, plast0
            enddo
         endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if( side. eq. 0) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then                    
                     stencil(WW,ie0,ic1) = stencil(WW,ie0,ic1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(EE,ie0,ic1) = stencil(EE,ie0,ic1)
     &                                   +stencil(WW,ie0,ic1)/3.0
                     stencil(WW,ie0,ic1) = 8.0*stencil(WW,ie0,ic1)/3.0
                  endif
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then                    
                     stencil(EE,ie0-1,ic1) = stencil(EE,ie0-1,ic1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(WW,ie0-1,ic1)=stencil(WW,ie0-1,ic1)
     &                                   +stencil(EE,ie0-1,ic1)/3.0
                     stencil(EE,ie0-1,ic1)= 8.0*stencil(EE,ie0-1,ic1)/3.0
                  endif
               enddo
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if( side .eq. 0) then
               do ic0 = pfirst0, plast0
                  if(extrapOrder.eq.1) then                    
                     stencil(SS,ic0,ie1) = stencil(SS,ic0,ie1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(NN,ic0,ie1)=stencil(NN,ic0,ie1)
     &                                 +stencil(SS,ic0,ie1)/3.0
                     stencil(SS,ic0,ie1)=8.0*stencil(SS,ic0,ie1)/3.0
                  endif
               enddo
            else
               do ic0 = pfirst0, plast0
                  if(extrapOrder.eq.1) then                    
                     stencil(NN,ic0,ie1-1) = stencil(NN,ic0,ie1-1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(SS,ic0,ie1-1)=stencil(SS,ic0,ie1-1)
     &                                   +stencil(NN,ic0,ie1-1)/3.0
                     stencil(NN,ic0,ie1-1)=8.0*stencil(NN,ic0,ie1-1)/3.0
                  endif
               enddo
            endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         factor = neufactor
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if(side .eq. 0) then
               do ic1 = pfirst1, plast1
                  stencil(WW,ie0,ic1) = 0.0d0
               enddo
            else
               do ic1 = pfirst1, plast1
                  stencil(EE,ie0-1,ic1) = 0.0d0
               enddo                  
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if(side .eq. 0) then
               do ic0 = pfirst0, plast0
                  stencil(SS,ic0,ie1) = 0.0d0
               enddo
            else
               do ic0 = pfirst0, plast0
                  stencil(NN,ic0,ie1-1) = 0.0d0
               enddo
            endif
         endif
      endif      
c     
      return
      end
c
      recursive subroutine adjcellpoissonoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
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
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side, bdrytype,
     &  extrapOrder
      REAL
     &  dirfactor, neufactor
      REAL 
     &  dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:4, CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1
      REAL factor
      REAL h
c
c***********************************************************************
c
c for now we will only do the dir=0 robin boundary conditions
c we will also assume homogenous boundary conditions for now

      if (bdrytype.eq.ROBIN) then
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            h=dx(0)
            if (side .eq. 0 ) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                    factor=(2.0*h)/(4.0+h)
                    stencil(WW,ie0,ic1)=factor*stencil(WW,ie0,ic1)
                  else if(extrapOrder.eq.2) then
                    factor=beta/(h*(3.0*h+16.0))
                    stencil(WW,ie0,ic1)=-9.0*factor
                    stencil(EE,ie0,ic1)=stencil(EE,ie0,ic1)-factor 
                  endif                 
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then
                    factor=2.0*h/(4.0+h)
                    stencil(EE,ie0-1,ic1)=factor*stencil(EE,ie0-1,ic1)
                  else if(extrapOrder.eq.2) then
                    factor=beta/(h*(3.0*h+16.0))
                    stencil(EE,ie0-1,ic1)=-9.0*factor
                    stencil(WW,ie0-1,ic1)=stencil(WW,ie0-1,ic1)-factor
                  endif
               enddo
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            do ic0 = pfirst0, plast0
            enddo
         endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if( side. eq. 0) then
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then                    
                     stencil(WW,ie0,ic1) = stencil(WW,ie0,ic1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(EE,ie0,ic1) = stencil(EE,ie0,ic1)
     &                                   +stencil(WW,ie0,ic1)/3.0
                     stencil(WW,ie0,ic1) = 8.0*stencil(WW,ie0,ic1)/3.0
                  endif
               enddo
            else
               do ic1 = pfirst1, plast1
                  if(extrapOrder.eq.1) then                    
                     stencil(EE,ie0-1,ic1) = stencil(EE,ie0-1,ic1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(WW,ie0-1,ic1)=stencil(WW,ie0-1,ic1)
     &                    +stencil(EE,ie0-1,ic1)/3.0
                     stencil(EE,ie0-1,ic1)= 8.0*stencil(EE,ie0-1,ic1)/3.0
                  endif
               enddo
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if( side .eq. 0) then
               do ic0 = pfirst0, plast0
                  if(extrapOrder.eq.1) then                    
                     stencil(SS,ic0,ie1) = stencil(SS,ic0,ie1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(NN,ic0,ie1)=stencil(NN,ic0,ie1)
     &                                 +stencil(SS,ic0,ie1)/3.0
                     stencil(SS,ic0,ie1)=8.0*stencil(SS,ic0,ie1)/3.0
                  endif
               enddo
            else
               do ic0 = pfirst0, plast0
                  if(extrapOrder.eq.1) then                    
                     stencil(NN,ic0,ie1-1) = stencil(NN,ic0,ie1-1)*factor
                  else if(extrapOrder.eq.2) then
                     stencil(SS,ic0,ie1-1)=stencil(SS,ic0,ie1-1)
     &                                   +stencil(NN,ic0,ie1-1)/3.0
                     stencil(NN,ic0,ie1-1)=8.0*stencil(NN,ic0,ie1-1)/3.0
                  endif
               enddo
            endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         factor = neufactor
         if (direction.eq.0) then
            ie0 = pfirst0-side+1
            if(side .eq. 0) then
               do ic1 = pfirst1, plast1
                  stencil(WW,ie0,ic1) = 0.0d0
               enddo
            else
               do ic1 = pfirst1, plast1
                  stencil(EE,ie0-1,ic1) = 0.0d0
               enddo                  
            endif
         elseif (direction.eq.1) then
            ie1 = pfirst1-side+1
            if(side .eq. 0) then
               do ic0 = pfirst0, plast0
                  stencil(SS,ic0,ie1) = 0.0d0
               enddo
            else
               do ic0 = pfirst0, plast0
                  stencil(NN,ic0,ie1-1) = 0.0d0
               enddo
            endif
         endif
      endif
c     
      return
      end
c
c
      recursive subroutine adjcelldiffusioncfoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  cfirst0,cfirst1,clast0,clast1,
     &  r,
     &  direction, side,
     &  interporder,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  cfirst0,cfirst1,clast0,clast1,
     &  direction, side
      integer r
      integer interporder
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1
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
               do ic1 = cfirst1, clast1
                  stencil(WW,ie0,ic1)=2.0*stencil(WW,ie0,ic1)/3.0
               enddo
            else if(interporder .eq. 2) then
               do ic1 = cfirst1, clast1
                  stencil(EE,ie0,ic1) = stencil(EE,ie0,ic1)
     &                           -stencil(WW,ie0,ic1)*(dr-1.0)/(dr+3.0)
                  stencil(WW,ie0,ic1)=
     &                       8.0*stencil(WW,ie0,ic1)/((dr+1.0)*(dr+3.0))
               enddo
            endif
         else 
            if(interporder .eq. 1) then
               do ic1 = cfirst1, clast1
                  stencil(EE,ie0-1,ic1)=2.0*stencil(EE,ie0-1,ic1)/3.0
               enddo
            else if(interporder .eq. 2) then
               do ic1 = cfirst1, clast1
                  stencil(WW,ie0-1,ic1) = stencil(WW,ie0-1,ic1)
     &                           -stencil(EE,ie0-1,ic1)*(dr-1.0)/(dr+3.0)
                  stencil(EE,ie0-1,ic1)=
     &                      8.0*stencil(EE,ie0-1,ic1)/((dr+1.0)*(dr+3.0))
               enddo
            endif
         endif
      elseif (direction.eq.1) then
         ie1 = cfirst1-side+1
         if(side.eq.0) then
            if(interporder .eq. 1) then
               do ic0 = cfirst0, clast0
                  stencil(SS,ic0,ie1)=2.0*stencil(SS,ic0,ie1)/3.0
               enddo
            else if(interporder .eq. 2) then
               do ic0 = cfirst0, clast0
                  stencil(NN,ic0,ie1) = stencil(NN,ic0,ie1)
     &                 -stencil(SS,ic0,ie1)*(dr-1.0)/(dr+3.0)
                  stencil(SS,ic0,ie1)=
     &                    8.0*stencil(SS,ic0,ie1)/((dr+1.0)*(dr+3.0))
               enddo
            endif
         else
            if(interporder .eq. 1) then
               do ic0 = cfirst0, clast0
                  stencil(NN,ic0,ie1-1)=2.0*stencil(NN,ic0,ie1-1)/3.0
               enddo
            else if(interporder .eq. 2) then
               do ic0 = cfirst0, clast0
                  stencil(SS,ic0,ie1-1) = stencil(SS,ic0,ie1-1)
     &                 -stencil(NN,ic0,ie1-1)*(dr-1.0)/(dr+3.0)
                  stencil(NN,ic0,ie1-1)=
     &                      8.0*stencil(NN,ic0,ie1-1)/((dr+1.0)*(dr+3.0))
               enddo
            endif
         endif
      endif
      return
      end
c
c
      recursive subroutine readjcelldiffusionoffdiag2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer ic0,ic1,ie0,ie1
c
c***********************************************************************
c

      if (direction.eq.0) then
         ie0 = pfirst0-side+1
         if(side .eq. 0) then
            do ic1 = pfirst1, plast1
               stencil(WW,ie0,ic1) = 0.0
            enddo
         else
            do ic1 = pfirst1, plast1
               stencil(EE,ie0-1,ic1) = 0.0
            enddo
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1-side+1
         if(side .eq. 0) then
            do ic0 = pfirst0, plast0
               stencil(SS,ic0,ie1) = 0.0
            enddo
         else
            do ic0 = pfirst0, plast0
               stencil(NN,ic0,ie1-1) = 0.0
            enddo
         endif
      endif
c
      return
      end
c
c
      recursive subroutine adjcelldiffusioncfbdryrhs2d(
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  pfirst0,pfirst1,plast0,plast1,
     &  direction, side,
     &  sgcw,
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
      integer sgcw
      REAL stencil(0:4,CELL2d(ifirst,ilast,sgcw))
      integer gcw
      REAL
     &  u(CELL2d(ifirst,ilast,gcw)),
     &  rhs(CELL2d(ifirst,ilast,0))
      integer ic0,ic1,ie0,ie1
c
c***********************************************************************
c

      if (direction.eq.0) then
         ie0 = pfirst0+1-(2*side)
         if(side.eq.0) then
            do ic1 = pfirst1, plast1
               rhs(ie0,ic1) = rhs(ie0,ic1) -
     &              stencil(WW,ie0,ic1)*u(ie0-1,ic1)
            enddo
         else
            do ic1 = pfirst1, plast1
               rhs(ie0,ic1) = rhs(ie0,ic1) -
     &              stencil(EE,ie0,ic1)*u(ie0+1,ic1)
            end do
         endif
      elseif (direction.eq.1) then
         ie1 = pfirst1+1-(2*side)
         if(side.eq.0) then
            do ic0 = pfirst0, plast0
               rhs(ic0,ie1) = rhs(ic0,ie1)-
     &              stencil(SS,ic0,ie1)*u(ic0,ie1-1)
            enddo
         else
            do ic0 = pfirst0, plast0
               rhs(ic0,ie1) = rhs(ic0,ie1)-
     &              stencil(NN,ic0,ie1)*u(ic0,ie1+1)
            enddo
         endif
      endif
      return
      end
