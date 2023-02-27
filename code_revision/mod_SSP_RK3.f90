module mod_SSP_RK3
  use constants
  use parameters
  use global_variables
  use mod_FE


  implicit none

contains

  subroutine SSP_RK3_(Uin, Uout)

    real(PR), intent(inout) , dimension(4,lb:le, nb:ne) :: Uin
    real(PR), intent(out)   , dimension(4,lb:le, nb:ne) :: Uout

    real(PR),              dimension(4,lb:le, nb:ne) :: U1
    real(PR),              dimension(4,lb:le, nb:ne) :: U2

    t_rk = t
    count_RK = 0
    call Forward_Euler(Uin,U1 )
    count_RK = count_RK + count_FE

    t_rk = t + dt

    call Forward_Euler(U1 ,U2 )
    count_RK = count_RK + count_FE


    U2 = (0.75)*Uin + 0.25*U2
    t_rk = t + 0.5*dt

    call Forward_Euler(U2 ,Uout)
    count_RK = count_RK + count_FE


    Uout = (1./3)*Uin + (2./3)*Uout

    count_FE = count_RK /3


  end subroutine SSP_RK3_

end module mod_SSP_RK3
