module mod_GP_SE
  use constants
  use parameters
  use global_variables
  use physics

  implicit none

contains

  subroutine GP_SE_(Uin)

    real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

    integer                         :: l, n, k, i,j
    real, dimension(4,sz_cross) :: q
    real, dimension(4,sz_sphere) :: q_sp

    if (cross_stencil) then

      do n = 0, nf+1
        do l = 0, lf+1

          do k = 1, sz_cross
            q(:,k) = Uin(:,l+ixiy(mord,k,1), n+ixiy(mord,k,2))
          end do

          do j = 1, ngp
            do i = iL,iB

              do k = rho, ener
                Uh(k,i,j,l,n) = dot_product(zT(mord,1: sz_cross,i,j),q(k,1: sz_cross))
              end do

            end do
          end do

        end do
      end do

    else if (sphere_stencil) then

      do n = 0, nf+1
        do l = 0, lf+1

          do k = 1, sz_sphere
            q_sp(:,k) = Uin(:,l+ixiy_sp(mord,k,1), n+ixiy_sp(mord,k,2))
          end do

          do j = 1, ngp
            do i = iL,iB

              do k = rho, ener
                Uh(k,i,j,l,n) = dot_product(zT_sp(mord,1: sz_sphere,i,j),q_sp(k,1: sz_sphere))
              end do

            end do
          end do

        end do
      end do

    else
      print*, " no stencil selected"
      stop
    end if





  end subroutine
end module
