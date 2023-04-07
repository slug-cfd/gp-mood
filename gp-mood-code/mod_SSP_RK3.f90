module mod_SSP_RK3
   use constants
   use parameters
   use secondary_parameters
   use global_variables
   use mod_FE


   implicit none

contains

   subroutine SSP_RK3_(Uin, Uout)

      real(PR), intent(inout) , dimension(4,lb:le, nb:ne) :: Uin
      real(PR), intent(out)   , dimension(4,lb:le, nb:ne) :: Uout

      real(PR),              dimension(4,lb:le, nb:ne) :: U1
      real(PR),              dimension(4,lb:le, nb:ne) :: U2

      last_RK_stage=.false.

      t_rk = t
      count_detected_cell_RK = 0
      count_NN_PAD_RK = 0
      call Forward_Euler(Uin,U1, .true. )
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell
      count_NN_PAD_RK = count_NN_PAD_RK + count_NN_PAD


      t_rk = t + dt

      call Forward_Euler(U1 ,U2, .false. )
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell
      count_NN_PAD_RK = count_NN_PAD_RK + count_NN_PAD

      U2 = (0.75)*Uin + 0.25*U2
      t_rk = t + 0.5*dt

      last_RK_stage=.true.
      call Forward_Euler(U2 ,Uout, .false.)
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell
      count_NN_PAD_RK = count_NN_PAD_RK + count_NN_PAD

      Uout = (1./3)*Uin + (2./3)*Uout

      count_detected_cell = count_detected_cell_RK /3
      count_NN_PAD_RK = count_NN_PAD_RK/3



   end subroutine SSP_RK3_

end module mod_SSP_RK3
