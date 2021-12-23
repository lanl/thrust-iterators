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
c  File:        celldiffusionstencilcoeffs1d.m4
c  Package:     SAMRSolvers
c  Copyright:   (c) 1997-2001 The Regents of the University of California
c  Release:     $Name$
c  Revision:    $Revision: 2727 $
c  Modified:    $Date: 2006-06-22 15:52:36 -0600 (Thu, 22 Jun 2006) $
c  Description: F77 routines that compute matrix entries for 1d cell centered diffusion operator
c
define(NDIM,1)dnl
define(REAL,`double precision')dnl
define(PP,0)dnl
define(WW,1)dnl
define(EE,2)dnl
define(DIRICHLET,0)
define(NEUMANN,1)
define(ROBIN,4)
include(pdat_m4arrdim1d.i)dnl
c
c
      recursive subroutine celldiffusionoffdiag1d(
     &  ifirst0,ilast0,
     &  bilo0,bihi0,
     &  dx,
     &  beta,
     &  b0,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  bilo0,bihi0
      REAL
     &  dx(0:NDIM-1),beta,
     &  b0(FACE1d(bilo,bihi,0))
      integer sgcw

      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))

      integer ie0
      REAL d0
      integer e,w
c
c***********************************************************************
c
c using face data for b0,b1 will slow down the creation
c of the stencils significantly. we should move to using
c side data for b0,b1
      
      w=0
      e=1
      
      d0 = -beta / (dx(0)*dx(0))
      
      do ie0 = ifirst0, ilast0
         stencil(WW, ie0) = d0*b0(ie0  ) 
         stencil(EE, ie0) = d0*b0(ie0+1) 
      enddo
      
      return
      end
c     
      recursive subroutine cellpoissonoffdiag1d(
     &  ifirst0,ilast0,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0
      REAL dx(0:NDIM-1),beta
      integer sgcw

      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))

      integer ie0
      REAL d0
      integer e,w
c
c***********************************************************************
c
      w=0
      e=1

      d0 = -beta / (dx(0)*dx(0))
      
      do ie0 = ifirst0, ilast0
         stencil(WW, ie0) = d0
         stencil(EE, ie0) = d0
      enddo
      
      return
      end
c
c
      recursive subroutine celldiffusionv1diag1d(
     &  ifirst0,ilast0,
     &  ailo0,aihi0,
     &  alpha,
     &  a,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  ailo0,aihi0
      REAL
     &  alpha,
     &  a(CELL1d(ailo,aihi,0))
      integer sgcw
      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))
      integer ic0
      integer e,w,p
c
c***********************************************************************
c
      w=0
      e=1
      p=2

      do ic0 = ifirst0, ilast0
         stencil(PP, ic0) = -( stencil(WW, ic0) 
     &        + stencil(EE, ic0))
     &        + alpha * a(ic0)
      enddo
c
      return
      end
c
c     
      recursive subroutine celldiffusionv2diag1d(
     &  ifirst0,ilast0,
     &  alpha,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      REAL alpha
      integer sgcw
      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))
      integer ic0
      integer e,w,p
c
c***********************************************************************
c
      w=0
      e=1
      p=2

      do ic0 = ifirst0, ilast0
         stencil(PP, ic0) = -(stencil(WW, ic0)+stencil(EE, ic0))+alpha
      enddo
c
      return
      end
c
c     
      recursive subroutine cellpoissondiag1d(
     &  ifirst0,ilast0,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer ifirst0,ilast0
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ic0
      integer e,w,p
c     
c***********************************************************************
c
      w=0
      e=1
      p=2

      do ic0 = ifirst0, ilast0
         stencil(PP, ic0) = -( stencil(WW, ic0) + stencil(EE, ic0) )
      enddo
c
      return
      end
c
c
      recursive subroutine adjcelldiffusiondiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  beta,
     &  b0,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side, bdrytype
      integer extrapOrder
      REAL dx(0:NDIM-1), beta
      REAL b0(FACE1d(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0,edge
      
      REAL factor
      REAL b, h
      integer p
c
c***********************************************************************
c
      p=2

      if (bdrytype.eq.ROBIN) then
         ie0 = pfirst0-2*side+1
         edge= pfirst0-side+1
         b=0.25
         h=dx(0)
         if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
            else if (extrapOrder.eq.2) then
               factor=beta*b0(edge)/h*(3.0*h+16.0*b0(edge))
               stencil(PP,ie0)=stencil(PP,ie0)-factor
            endif
         else
            if(extrapOrder.eq.1) then
            else if (extrapOrder.eq.2) then
               factor=beta*b0(edge)/h*(3.0*h+16.0*b0(edge))
               stencil(PP,ie0)=stencil(PP,ie0)-factor
            endif
         endif
      else
         if (bdrytype.eq.NEUMANN) then
            ie0 = pfirst0-side+1
         elseif (bdrytype.eq.DIRICHLET) then
         endif         
      endif

      return
      end

      recursive subroutine adjcellpoissondiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
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
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side, bdrytype
      integer extrapOrder
      REAL 
     &  dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0,edge
      
      REAL factor
      REAL b, h
      integer p
c
c***********************************************************************
c
      p=2

      if (bdrytype.eq.ROBIN) then
         ie0 = pfirst0-2*side+1
         edge= pfirst0-side+1
         b=0.25
         h=dx(0)
         if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
            else if (extrapOrder.eq.2) then
               factor=beta/h*(3.0*h+16.0)
               stencil(PP,ie0)=stencil(PP,ie0)-factor
            endif
         else
            if(extrapOrder.eq.1) then
            else if (extrapOrder.eq.2) then
               factor=beta/h*(3.0*h+16.0)
               stencil(PP,ie0)=stencil(PP,ie0)-factor
            endif
         endif
      else
         if (bdrytype.eq.NEUMANN) then
            ie0 = pfirst0-side+1
         elseif (bdrytype.eq.DIRICHLET) then
         endif         
      endif

      return
      end

      recursive subroutine adjcelldiffusioncfdiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  r,
     &  side,
     &  interporder,
     &  dx,
     &  beta,
     &  b0,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  interporder
      integer r
      REAL 
     &  dx(0:NDIM-1), beta
      REAL b0(FACE1d(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0,edge
      
      REAL factor
      REAL h

      REAL dr,dr1

      integer p

c r is the refinement ratio, convert to double precision      
      p=2
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      ie0 = pfirst0-2*side+1
      edge= pfirst0-side+1
      h=dx(0)
      if (interporder .eq. 1) then
         factor=beta*b0(edge)/(3.0*(h**2))
         stencil(PP,ie0)=stencil(PP,ie0)-factor
      else if (interporder .eq. 2) then
         factor=dr1*beta*b0(edge)/(h**2)
         stencil(PP,ie0)=stencil(PP,ie0)-factor
      endif
      
      return
      end
c
      recursive subroutine adjcellpoissoncfdiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  r,
     &  side,
     &  interporder,
     &  dx,
     &  beta,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  interporder
      integer r
      REAL dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0,edge
      
      REAL factor
      REAL h
      REAL dr,dr1
      integer p
c r is the refinement ratio, convert to double precision      
      p=2
      dr = dfloat(r)
      dr1=2.0*(dr-1.0)/(dr+1.0)
c
c***********************************************************************
c
      ie0 = pfirst0-2*side+1
      edge= pfirst0-side+1
      h=dx(0)
      if (interporder .eq. 1) then
         factor=beta/(3.0*(h**2))
         stencil(PP,ie0)=stencil(PP,ie0)-factor
      else if (interporder .eq. 2) then
         factor=dr1*beta/(h**2)
         stencil(PP,ie0)=stencil(PP,ie0)-factor
      endif
      
      return
      end
c
c
      recursive subroutine adjcelldiffusionoffdiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  bdrytype,
     &  extrapOrder,
     &  dx,
     &  dirfactor, neufactor, 
     &  beta,
     &  b0,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side, bdrytype,
     &  extrapOrder
      REAL dirfactor, neufactor
      REAL dx(0:NDIM-1), beta
      REAL b0(FACE1d(ifirst,ilast,0))
      integer sgcw
      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))
      integer ie0
      REAL factor
      REAL h
      integer e,w
c
c***********************************************************************
c
      w=0
      e=1

      if (bdrytype.eq.ROBIN) then
         ie0 = pfirst0-side+1
         h=dx(0)
         if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
               factor=(2.0*h)/(4.0*b0(ie0)+h)
               stencil(WW,ie0)=factor*stencil(WW,ie0)
            else if(extrapOrder.eq.2) then
               factor=beta*b0(ie0)/(h*(3.0*h+16.0*b0(ie0)))
               stencil(WW,ie0)=-9.0*factor
               stencil(EE,ie0)=stencil(EE,ie0)-factor 
            endif                 
         else
            if(extrapOrder.eq.1) then
               factor=2.0*h/(4.0*b0(ie0)+h)
               stencil(EE,ie0-1)=factor*stencil(EE,ie0-1)
            else if(extrapOrder.eq.2) then
               factor=beta*b0(ie0)/(h*(3.0*h+16.0*b0(ie0)))
               stencil(EE,ie0-1)=-9.0*factor
               stencil(WW,ie0-1)=stencil(WW,ie0-1)-factor
            endif
         endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         ie0 = pfirst0-side+1
         if( side. eq. 0) then
            if(extrapOrder.eq.1) then                    
               stencil(WW,ie0) = stencil(WW,ie0)*factor
            else if(extrapOrder.eq.2) then
               stencil(EE,ie0) = stencil(EE,ie0)
     &              +stencil(WW,ie0)/3.0
               stencil(WW,ie0) = 8.0*stencil(WW,ie0)/3.0
            endif
         else
            if(extrapOrder.eq.1) then                    
               stencil(EE,ie0-1) = stencil(EE,ie0-1)*factor
            else if(extrapOrder.eq.2) then
               stencil(WW,ie0-1)=stencil(WW,ie0-1)
     &              +stencil(EE,ie0-1)/3.0
               stencil(EE,ie0-1)= 8.0*stencil(EE,ie0-1)/3.0
            endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         factor = neufactor
         ie0 = pfirst0-side+1
         if(side .eq. 0) then
            stencil(WW,ie0) = 0.0d0
         else
            stencil(EE,ie0-1) = 0.0d0
         endif
      endif      
c     
      return
      end
c
      recursive subroutine adjcellpoissonoffdiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
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
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side, bdrytype,
     &  extrapOrder
      REAL dirfactor, neufactor
      REAL dx(0:NDIM-1), beta
      integer sgcw
      REAL stencil(0:2, CELL1d(ifirst,ilast,sgcw))
      integer ie0
      REAL factor
      REAL h
      integer e,w
c
c***********************************************************************
c
c for now we will only do the dir=0 robin boundary conditions
c we will also assume homogenous boundary conditions for now
      w=0
      e=1

      if (bdrytype.eq.ROBIN) then
         ie0 = pfirst0-side+1
         h=dx(0)
         if (side .eq. 0 ) then
            if(extrapOrder.eq.1) then
               factor=(2.0*h)/(4.0+h)
               stencil(WW,ie0)=factor*stencil(WW,ie0)
            else if(extrapOrder.eq.2) then
               factor=beta/(h*(3.0*h+16.0))
               stencil(WW,ie0)=-9.0*factor
               stencil(EE,ie0)=stencil(EE,ie0)-factor 
            endif                 
         else
            if(extrapOrder.eq.1) then
               factor=2.0*h/(4.0+h)
               stencil(EE,ie0-1)=factor*stencil(EE,ie0-1)
            else if(extrapOrder.eq.2) then
               factor=beta/(h*(3.0*h+16.0))
               stencil(EE,ie0-1)=-9.0*factor
               stencil(WW,ie0-1)=stencil(WW,ie0-1)-factor
            endif
         endif
      else if (bdrytype.eq.DIRICHLET) then
         factor=dirfactor
         ie0 = pfirst0-side+1
         if( side. eq. 0) then
            if(extrapOrder.eq.1) then                    
               stencil(WW,ie0) = stencil(WW,ie0)*factor
            else if(extrapOrder.eq.2) then
               stencil(EE,ie0) = stencil(EE,ie0)+stencil(WW,ie0)/3.0
               stencil(WW,ie0) = 8.0*stencil(WW,ie0)/3.0
            endif
         else
            if(extrapOrder.eq.1) then                    
               stencil(EE,ie0-1) = stencil(EE,ie0-1)*factor
            else if(extrapOrder.eq.2) then
               stencil(WW,ie0-1)=stencil(WW,ie0-1)+stencil(EE,ie0-1)/3.0
               stencil(EE,ie0-1)= 8.0*stencil(EE,ie0-1)/3.0
            endif
         endif
      else if (bdrytype.eq.NEUMANN) then
         factor = neufactor
         ie0 = pfirst0-side+1
         if(side .eq. 0) then
            stencil(WW,ie0) = 0.0d0
         else
            stencil(EE,ie0-1) = 0.0d0
         endif
      endif
c     
      return
      end
c
c
      recursive subroutine adjcelldiffusioncfoffdiag1d(
     &  ifirst0,ilast0,
     &  cfirst0,clast0,
     &  r,
     &  side,
     &  interporder,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  cfirst0,clast0,
     &  side
      integer r
      integer interporder
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0
      integer offset
      REAL dr
      integer e,w

c r is the refinement ratio, convert to double precision      
      w=0
      e=1

      dr = dfloat(r)
c     
c***********************************************************************
      offset=1-2*side
      ie0 = cfirst0-side+1
      if(side.eq.0) then
         if(interporder .eq. 1) then
            stencil(WW,ie0)=2.0*stencil(WW,ie0)/3.0
         else if(interporder .eq. 2) then
            stencil(EE,ie0) = stencil(EE,ie0)
     &                      -stencil(WW,ie0)*(dr-1.0)/(dr+3.0)
            stencil(WW,ie0)= 8.0*stencil(WW,ie0)/((dr+1.0)*(dr+3.0))
         endif
      else 
         if(interporder .eq. 1) then
            stencil(EE,ie0-1)=2.0*stencil(EE,ie0-1)/3.0
         else if(interporder .eq. 2) then
            stencil(WW,ie0-1) = stencil(WW,ie0-1)
     &                        -stencil(EE,ie0-1)*(dr-1.0)/(dr+3.0)
            stencil(EE,ie0-1)= 8.0*stencil(EE,ie0-1)/((dr+1.0)*(dr+3.0))
         endif
      endif

      return
      end
c
c
      recursive subroutine readjcelldiffusionoffdiag1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  sgcw,
     &  stencil)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer ie0
      integer e,w
c
c***********************************************************************
c
      w=0
      e=1

      ie0 = pfirst0-side+1
      if(side .eq. 0) then
         stencil(WW,ie0) = 0.0
      else
         stencil(EE,ie0-1) = 0.0
      endif
c
      return
      end
c
c
      recursive subroutine adjcelldiffusioncfbdryrhs1d(
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side,
     &  sgcw,
     &  stencil,
     &  gcw,
     &  u, rhs)
c***********************************************************************
      implicit none
c
      integer
     &  ifirst0,ilast0,
     &  pfirst0,plast0,
     &  side
      integer sgcw
      REAL stencil(0:2,CELL1d(ifirst,ilast,sgcw))
      integer gcw
      REAL
     &  u(CELL1d(ifirst,ilast,gcw)),
     &  rhs(CELL1d(ifirst,ilast,0))
      integer ie0
      integer e,w
c
c***********************************************************************
c
      w=0
      e=1

      ie0 = pfirst0+1-(2*side)
      if(side.eq.0) then
         rhs(ie0) = rhs(ie0)-stencil(WW,ie0)*u(ie0-1)
      else
         rhs(ie0) = rhs(ie0)-stencil(EE,ie0)*u(ie0+1)
      endif

      return
      end
