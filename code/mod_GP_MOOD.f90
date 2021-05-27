module mod_GP_MOOD
  use constants
  use parameters
  use global_variables
  use physics
  use GP_init
  implicit none

contains

  subroutine GP_MOOD_(Uin)

    real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

    integer                     :: l, n, k, i,j
    real(PR), dimension(4,sz_cross)  :: q
    real(PR), dimension(4,sz_sphere) :: q_sp

    logical                     :: reconstruction, done
    integer                     :: ord

    real, dimension(4) :: u


    do n = 0, nf+1
       do l = 0, lf+1

          q    = 0.
          q_sp = 0.
          done = .false.

          do i = iL,iB

             reconstruction = .false.

             if ((i==iL).and.(DetFace_x(l-1,n))) then
                ord = min( CellGPO(l-1,n  ), CellGPO(l,n))
                reconstruction = .true.
             else if ((i==iR).and.(DetFace_x(l,n))) then
                ord = min( CellGPO(l+1,n  ), CellGPO(l,n))
                reconstruction = .true.
             else if ((i==iB).and.(DetFace_y(l,n-1))) then
                ord = min( CellGPO(l,n -1), CellGPO(l,n))
                reconstruction = .true.
             else if ((i==iT).and.(DetFace_y(l,n))) then
                ord = min( CellGPO(l,n+1  ), CellGPO(l,n))
                reconstruction = .true.
             end if


             if (reconstruction .eqv. .true.) then

                if (done .eqv. .false.) then
                   if ( cross_stencil ) then
                      do j = 1, sz_cross
                         q(:,j) = Uin(:,l+ixiy(mord,j,1),n+ixiy(mord,j,2))
                      end do
                   else
                      do j = 1, sz_sphere
                         q_sp(:,j) = Uin(:,l+ixiy_sp(mord,j,1),n+ixiy_sp(mord,j,2))
                      end do
                   end if


                   done = .true.

                end if


                do j = 1, ngp

                   do k = rho, ener
                      if ( cross_stencil ) then

                         Uh(k,i,j,l,n) = dot_product( zT(ord,1:stcl_sz(ord),i,j),q(k,1:stcl_sz(ord)) )
                      else
                         Uh(k,i,j,l,n) = dot_product( zT_sp(ord,1:stcl_sz(ord),i,j),q_sp(k,1:stcl_sz(ord)))

                      end if
                   end do

                end do


             end if

          end do

       end do
    end do


  end subroutine GP_MOOD_

end module mod_GP_MOOD
