module mod_subroutine_NN_MOOD
   use constants
   use parameters
   use global_variables
   use physics
   use BC
   use GP_init
   implicit none

contains


   subroutine NN_DETECTION(Uin,Uout)

      real(PR), dimension(4,lb:le, nb:ne ),intent(in   ) :: Uin
      real(PR), dimension(4,lb:le, nb:ne ),intent(inout) :: Uout

      logical , dimension(1:lf, 1:nf)                 :: decrease

      real(PR)                                        :: p

      integer :: l, n, count, count_PAD


      MOOD_finished = .true.
      decrease      = .false.

      count = 0
      count_PAD = 0

      call Boundary_C(Uout)


      do n = 1, nf
         do l = 1, lf

            if ((DetCell(l,n)).and.(CellGPO(l,n) > 1)) then

               if ((Uout(rho,l,n) <= 0.).or.(ISNAN(Uout(rho,l,n)))) then
                  decrease(l,n) = .true.
               else

                  p = pressure(Uout(1:4,l,n))

                  if ((p <= 0.).or.(ISNAN(p))) then
                     decrease(l,n) = .true.
                  end if

               end if

            end if

            if (decrease(l,n) .eqv. .false.) then

               DetCell(l,n)       = .false.
               DetFace_y(l  ,n-1) = .false.
               DetFace_y(l  ,n  ) = .false.
               DetFace_x(l-1,n  ) = .false.
               DetFace_x(l  ,n  ) = .false.

            end if

         end do
      end do



      do n = 1, nf
         do l = 1, lf
            if (decrease(l,n)) then
               count = count + 1
               count_PAD = count_PAD+1

               MOOD_finished = .false.

               if (CellGPO(l,n) == 3)  then
                  CellGPO(l,n) = 1
               elseif (CellGPO(l,n) >  3)  then
                  CellGPO(l,n) = 3
               end if

               DetCell(l  ,n-1) = .true.
               DetCell(l-1,n  ) = .true.
               DetCell(l  ,n  ) = .true.
               DetCell(l+1,n  ) = .true.
               DetCell(l  ,n+1) = .true.

               DetFace_y(l  ,n-1) = .true.
               DetFace_y(l  ,n  ) = .true.
               DetFace_x(l  ,n  ) = .true.
               DetFace_x(l-1,n  ) = .true.

               Uout(1:4,l  ,n-1) = Uin(1:4,l  ,n-1)
               Uout(1:4,l-1,n  ) = Uin(1:4,l-1,n  )
               Uout(1:4,l  ,n  ) = Uin(1:4,l  ,n  )
               Uout(1:4,l+1,n  ) = Uin(1:4,l+1,n  )
               Uout(1:4,l  ,n+1) = Uin(1:4,l  ,n+1)

            end IF
         end do
      end do

      count_detected_cell = count_detected_cell + count
      count_NN_PAD= count_NN_PAD + count_PAD

   end subroutine NN_DETECTION

end module mod_subroutine_NN_MOOD
