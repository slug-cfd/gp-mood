module mod_HLLC

  use parameters
  use constants
  use physics

  implicit none

contains

  function HLLC_Flux(ul,ur,dir)result(F)
    !=========== HLL Flux ===========!

    !  Eleuterio F. Toro-Riemann Solvers and Numerical Methods for Fluid Dynamics_ A Practical Introduction, Third Edition (2009)
    ! Chap 10.4

    !============= Input(s) ==============!
    real(PR)   , dimension(4), intent(in) :: ul, ur
    integer              , intent(in)     :: dir
    !============= Output(s) =============!
    real(PR), dimension(4)                :: F

    !========== Local variables ==========!
    real(PR), dimension(4)                :: Fl, Fr
    real(PR), dimension(4)                :: UstarL, UstarR

    real(PR):: vmcl, vpcl, vmcr, vpcr, SL, SR
    real(PR):: uStar,  vvR, vvL, presL, presR, numerL,numerR,denomL,denomR, pstarL, pstarR, dstarl, dstarR


    real(PR)::  vx, vy, vx2, vy2, pres, c, other_velL, other_velR

    logical :: adm
    integer :: momdir, othermom


    ! !=========== = Instructions(l) ===========!


    !We compute the eingenvalues
    !Select the biggest and the smallest one

    adm = .true.

    if (dir == dir_x) then

       momdir   = momx
       othermom = momy

       call c2p_local(ul, vx, vy, vx2, vy2, pres, c, adm)
       vmcl = vx - c
       vpcl = vx + c
       vvL  = vx

       presL = pres
       other_velL = vy


       Fl   = Flux(ul,dir, vx, vy, vx2, vy2, pres, adm)

       call c2p_local(ur, vx, vy, vx2, vy2, pres, c, adm)
       vmcr = vx - c
       vpcr = vx + c
       vvR  = vx

       presR = pres
       other_velR = vy


       Fr   = Flux(ur,dir, vx, vy, vx2, vy2, pres, adm)

       SL = min(vmcl, vmcr)
       SR = max(vpcl, vpcr)


    else if (dir == dir_y) then

       momdir   = momy
       othermom = momx

       call c2p_local(ul, vx, vy, vx2, vy2, pres, c, adm)
       vmcl = vy - c
       vpcl = vy + c
       vvL  = vy

       presL = pres
       other_velL = vx



       Fl   = Flux(ul,dir, vx, vy, vx2, vy2, pres, adm)

       call c2p_local(ur, vx, vy, vx2, vy2, pres, c, adm)
       vmcr = vy - c
       vpcr = vy + c
       vvR  = vy

       presR = pres
       other_velR = vx


       Fr   = Flux(ur,dir, vx, vy, vx2, vy2, pres, adm)

       SL = min(vmcl, vmcr)
       SR = max(vpcl, vpcr)



    else
       print*, 'dir =/ x or y'
       stop
    end if

    if (adm) then
       ! Convenient parameters
       numerL = sL-vvL
       numerR = sR-vvR

       ! Symmetry fix from Numerical symmetry-preserving techniques for low-dissipation shock-capturing schemes
       ! Eqn 16
       uStar = ul(rho)*vvL*numerL - ur(rho)*vvR*numerR
       uStar = (presR - presL) + uStar
       uStar = uStar/(ul(rho)*numerL-ur(rho)*numerR)

       ! Convenient parameters
       denomL = sL-uStar
       denomR = sR-uStar

       ! rho
       dStarL = uL(rho)*numerL/denomL
       dStarR = uR(rho)*numerR/denomR

       ! Get pStar
       pStarL = uL(rho)*numerL*(uStar-vvL) + presL

       pStarR = pStarL ! concistency


       ! left and right star regions
       UstarL(rho)      =  1.
       UstarL(momdir)   = uStar
       UstarL(othermom) = other_velL
       UstarL(ENER)     = uL(ener)/uL(rho) + (uStar - vvL)*( uStar + presL/(ul(rho)*numerL) )

       UstarL = UstarL * dStarL

       UstarR(rho)      =  1.
       UstarR(momdir)   = uStar
       UstarR(othermom) = other_velR
       UstarR(ENER)     = uR(ener)/uR(rho) + (uStar - vvR)*( uStar + presR/(ur(rho)*numerR) )

       UstarR = UstarR * dStarR

       ! numerical flux
       if (sL >= 0.) then
          F(:) = FL(:)
       elseif ( (sL < 0.) .and. (uStar >= 0.) ) then
          F(:) = FL(:) + sL*(UstarL(:) - uL(:))
       elseif ( (uStar < 0.) .and. (sR >= 0.) ) then
          F(:) = FR(:) + sR*(UstarR(:) - uR(:))
       else
          F(:) = FR(:)
       endif

    else
       F(:) = 1./(vx-vx)
    end if

  end function HLLC_Flux


end module mod_HLLC
