module mod_SSP_RK2
   use constants
   use parameters
   use secondary_parameters
   use global_variables
   use mod_FE


   implicit none

contains

   subroutine SSP_RK2_(Uin, Uout)

      real(PR), intent(inout) , dimension(4,lb:le, nb:ne) :: Uin
      real(PR), intent(out), dimension(4,lb:le, nb:ne) :: Uout

      real(PR),              dimension(4,lb:le, nb:ne) :: U1

      count_detected_cell_RK = 0
      call Forward_Euler(Uin,U1, .true.  )

      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      call Forward_Euler(U1 ,Uout, .false.)
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      count_detected_cell = count_detected_cell_RK/2


      Uout = 0.5*(Uin+Uout)
   end subroutine

end module mod_SSP_RK2
