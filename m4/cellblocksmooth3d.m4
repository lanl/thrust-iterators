include(pdat_m4arrdim3d.i)dnl

define(NDIM,3)dnl
define(REAL,`double precision')dnl

c Create a dense matrix from stencil.
      recursive subroutine creatematrix(
     &  A,stencilSize,stencil,offset,
     &  bSizeX,bSizeY,bSizeZ)

      implicit none

      integer stencilSize,bSizeX,bSizeY,bSizeZ
      REAL A(0:bSizeX*bSizeY*bSizeZ-1,
     &       0:bSizeX*bSizeY*bSizeZ-1)
      REAL stencil(0:stencilSize-1,
     &    0:bSizeX-1,0:bSizeY-1,0:bSizeZ-1)
      integer offset(0:NDIM-1,0:stencilSize-1)

      integer i,j,k,ix,iy,iz
      integer s,idx,idxr

      do k=0,bSizeZ-1
         do j=0,bSizeY-1
            do i=0,bSizeX-1
               idx = k*bSizeY*bSizeX+j*bSizeX+i
               A(idx,idx) = stencil(0,i,j,k)
               do s=1,stencilSize
                  ix = i+offset(0,s)
                  iy = j+offset(1,s)
                  iz = k+offset(2,s)
                  if(ix.ge.0.and.ix.lt.bSizeX.and.
     &               iy.ge.0.and.iy.lt.bSizeY.and.
     &               iz.ge.0.and.iz.lt.bSizeZ) then
                     idxr = iz*bSizeY*bSizeX+iy*bSizeX+ix
                     A(idx,idxr) = stencil(s,i,j,k)
                  endif
               enddo
            enddo
         enddo
      enddo

      return
      end

c Solve a linear system Ax=b.
      recursive subroutine apply(
     & A,x,b,n)

      implicit none

      REAL A(0:n-1,0:n-1)
      REAL x(0:n-1),b(0:n-1),ipiv(0:n-1)
      integer n,error,one
      character trans

      trans='N'
      one = 1

c LU factorization.
      call dgetrf(n,n,A,n,ipiv,error);
      if(error.ne.0) then
         write (*,*) "cellblocksmooth3d::solve: Error
     &      when compute LU."
      endif

c Copy x to b.
      call dcopy(n,b,one,x,one);

c Solve.
      call dgetrs(trans,n,one,A,n,ipiv,x,n,error);
      if(error.ne.0) then
         write (*,*) "cellblocksmooth3d::solve: Error
     &      when solving."
      endif

      return
      end

      recursive subroutine cellblockjacobi3d(
     &  lo0,hi0,
     &  lo1, hi1,
     &  lo2, hi2,
     &  stencilSize,
     &  offset,
     &  stencil,
     &  flo0, fhi0,
     &  flo1, fhi1,
     &  flo2, fhi2,
     &  f,
     &  ulo0, uhi0,
     &  ulo1, uhi1,
     &  ulo2, uhi2,
     &  u,
     &  bDispX, bNumX, bSizeX,
     &  bDispY, bNumY, bSizeY,
     &  bDispZ, bNumZ, bSizeZ)

      implicit none

      integer lo0,lo1,lo2,hi0,hi1,hi2
      integer ulo0,ulo1,ulo2,uhi0,uhi1,uhi2
      integer flo0,flo1,flo2,fhi0,fhi1,fhi2
      integer stencilSize
      integer offset(0:NDIM-1,0:stencilSize-1)
      REAL stencil(0:stencilSize-1,CELL3d(lo,hi,0))
      REAL u(CELL3d(ulo,uhi,0))
      REAL f(CELL3d(flo,fhi,0))
      integer bNumX,bSizeX,bNumY,bSizeY,bNumZ,bSizeZ
      integer bDispX(0:bNumX-1)
      integer bDispY(0:bNumY-1)
      integer bDispZ(0:bNumZ-1)

      REAL r(CELL3d(lo,hi,0))
      integer i,j,k,s,io,jo,ko,bi,bj,bk,n
      REAL w
      REAL A(0:bSizeX*bSizeY*bSizeZ-1,
     &       0:bSizeX*bSizeY*bSizeZ-1)
      REAL br(0:bSizeX-1,0:bSizeY-1,0:bSizeZ-1)
      REAL update(0:bSizeX-1,0:bSizeY-1,0:bSizeZ-1)
      REAL bStencil(0:stencilSize-1,0:bSizeX-1,
     &              0:bSizeY-1,0:bSizeZ-1)

      n = bSizeX*bSizeY*bSizeZ
      w = 0.8D0

c we make the assumption that the diagonal entry for the stencil is
c always stored first in the stencil

c compute residual
      do k = lo2,hi2
         do j = lo1,hi1
            do i = lo0,hi0
               r(i,j,k) = f(i,j,k)
               do s=0,stencilSize-1
                  io=offset(0,s)
                  jo=offset(1,s)
                  ko=offset(2,s)
                  r(i,j,k) = r(i,j,k)
     &                      -stencil(s,i,j,k)*u(i+io,j+jo,k+ko)
               enddo
c This is pointwise Jacobi.
c               r(i,j,k) = r(i,j,k)/stencil(0,i,j,k)
            end do         
         end do
      end do

c      do k = lo2,hi2
c         do j = lo1,hi1
c            do i = lo0,hi0
c                u(i,j,k) = u(i,j,k)+w*r(i,j,k)
c            enddo
c         enddo
c      enddo

c perform solve for blocks
      do k = lo2+bDispZ(0),lo2+bDispZ(bNumZ-1),bSizeZ
         do j = lo1+bDispY(0),lo1+bDispY(bNumY-1),bSizeY
            do i = lo0+bDispX(0),lo0+bDispX(bNumX-1),bSizeX

c copy residual to a contiguous block
                br(0:bSizeX-1,0:bSizeY-1,0:bSizeZ-1)
     &          = r(i:i+bSizeX-1,j:j+bSizeY-1,k:k+bSizeZ-1)

c copy stencil to a contiguous block
                bStencil(0:stencilSize-1,0:bSizeX-1,
     &                   0:bSizeY-1,0:bSizeZ-1)
     &          = stencil(0:stencilSize-1,i:i+bSizeX-1,
     &                    j:j+bSizeY-1,k:k+bSizeZ-1)

c Solve for a block.
c Initialize A to be 0.
                A = 0
                call createMatrix(A,stencilSize,bStencil,offset,
     &              bSizeX,bSizeY,bSizeZ)
                call apply(A,update,br,n);

c copy residual to a contiguous block
                u(i:i+bSizeX-1,j:j+bSizeY-1,k:k+bSizeZ-1)
     &          = u(i:i+bSizeX-1,j:j+bSizeY-1,k:k+bSizeZ-1)
     &          + w*update(0:bSizeX-1,0:bSizeY-1,0:bSizeZ-1)

            enddo
         enddo
      enddo

      return
      end
