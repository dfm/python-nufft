
  subroutine wrap_dirft1d1(nj,xj,cj, iflag, ms,fk)
      implicit none
      integer, intent(in) :: nj, iflag, ms
      double precision, intent(in) :: xj(nj)
      complex*16, intent(in) :: cj(nj)
      complex*16, intent(out) :: fk(-ms/2:(ms-1)/2)
      call dirft1d1(nj,xj,cj, iflag, ms,fk)
  end subroutine wrap_dirft1d1

  ! subroutine wrap_nufft1d1(nj,xj,cj, iflag,eps, nk,sk,fk,ier)
  !     implicit none
  !     integer, intent(in) :: nj, iflag, nk
  !     double precision, intent(in) :: eps, xj(nj), sk(nk)
  !     complex*16, intent(in) :: cj(nj)
  !     integer, intent(out) :: ier
  !     complex*16, intent(out) :: fk(nk)
  !     call nufft1d1f90(nj,xj,cj, iflag,eps, nk,sk,fk,ier)
  ! end subroutine wrap_nufft1d1

  subroutine wrap_dirft1d3(nj,xj,cj, iflag, nk,sk,fk)
      implicit none
      integer, intent(in) :: nj, iflag, nk
      double precision, intent(in) :: xj(nj), sk(nk)
      complex*16, intent(in) :: cj(nj)
      complex*16, intent(out) :: fk(nk)
      call dirft1d3(nj,xj,cj, iflag, nk,sk,fk)
  end subroutine wrap_dirft1d3

  subroutine wrap_nufft1d3(nj,xj,cj, iflag,eps, nk,sk,fk,ier)
      implicit none
      integer, intent(in) :: nj, iflag, nk
      double precision, intent(in) :: eps, xj(nj), sk(nk)
      complex*16, intent(in) :: cj(nj)
      integer, intent(out) :: ier
      complex*16, intent(out) :: fk(nk)
      call nufft1d3f90(nj,xj,cj, iflag,eps, nk,sk,fk,ier)
  end subroutine wrap_nufft1d3
