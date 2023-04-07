module mod_SSP_RK4
   use constants
   use parameters
   use global_variables
   use mod_FE


   implicit none

contains

   subroutine SSP_RK4_(Uin, Uout)

      real(PR), intent(inout) , dimension(4,lb:le, nb:ne) :: Uin
      real(PR), intent(out), dimension(4,lb:le, nb:ne) :: Uout

      real(PR),              dimension(4,lb:le, nb:ne) :: U1, U2, U3, U4, FU3, FU4,Fl


      count_detected_cell_RK = 0
      call Forward_Euler(Uin, Fl, .true.)
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      U1 = Uin + c1*Fl
      call Forward_Euler(U1, Fl, .false. )
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      U2 = a20*Uin + a21*U1 + c2*Fl
      call Forward_Euler(U2, Fl, .false. )
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      U3 = a30*Uin + a32*U2 + c3*Fl
      call Forward_Euler(U3, FU3, .false.)
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      U4 = a40*Uin + a43*U3 + c4*FU3
      call Forward_Euler(U4, FU4, .false.)
      count_detected_cell_RK = count_detected_cell_RK + count_detected_cell

      count_detected_cell = count_detected_cell_RK /5

      Uout = f2*u2 + f3*U3 + ff3*FU3 + f4*u4 + ff4*FU4
   end subroutine

end module
