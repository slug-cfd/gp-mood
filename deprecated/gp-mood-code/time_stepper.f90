module time_stepper

  use parameters
  use constants
  use mod_FE
  use mod_SSP_RK2
  use mod_SSP_RK3
  use mod_SSP_RK4



contains

  subroutine time_stepping(Uin, Uout)

    real(PR), intent(inout), dimension(4,lb:le, nb:ne) :: Uin
    real(PR), intent(out  ), dimension(4,lb:le, nb:ne) :: Uout

    t_rk = t !Used for time dependant BC such as DMR
    if (time_method == FE     ) call Forward_Euler(Uin, Uout, .true.)
    if (time_method == SSP_RK2) call SSP_RK2_  (Uin, Uout)
    if (time_method == SSP_RK3) call SSP_RK3_  (Uin, Uout)
    if (time_method == SSP_RK4) call SSP_RK4_  (Uin, Uout)

  end subroutine

end module
