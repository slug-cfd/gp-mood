module mod_POL_MOOD
  use constants
  use parameters
  use global_variables
  use physics

  implicit none

contains

  subroutine POL_MOOD_(Uin)

    real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

    integer                     :: l, n, k, i,j
    real(PR), dimension(4,sz_cross) :: q
    logical                     :: reconstruction, done
    integer                     :: ord

    real, dimension(4) :: u


    do n = 0, nf+1
      do l = 0, lf+1

        q = 0.
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
              do j = 1, sz_cross
                q(:,j) = Uin(:,l+ixiy(mord,j,1),n+ixiy(mord,j,2))
              end do

              done = .true.

            end if


            do j = 1, ngp

              if ((ord == 3).and.(ngp /= 2)) then
                print*, 'This pol mood is not programmed'
                stop
              end if

              if ((ord == 5).and.(ngp /= 2)) then
                print*, 'This pol mood is not programmed'
                stop
              end if


              if (ord == 5) then

                do k = rho, ener
                  Uh(k,i,j,l,n) = dot_product( Pol_zT_o5(1:2*ord-1,i,j),q(k,1:2*ord-1) )
                end do

              else if (ord == 3) then

                do k = rho, ener
                  Uh(k,i,j,l,n) = dot_product( Pol_zT_o3(1:2*ord-1,i,j),q(k,1:2*ord-1) )
                end do

              else
                Uh(:,i,j,l,n) = Uin(:,l,n)
              end if

            end do


          end if

        end do

      end do
    end do


  end subroutine

end module
