module mod_unsplit

    use parameters
    use constants
    use physics
 
    implicit none
 
 contains
 
    function unsplit_flux(ul,ur,dir)result(F)
       !=========== HLL Flux ===========!
 
       !  Eleuterio F. Toro-Riemann Solvers and Numerical Methods for Fluid Dynamics_ A Practical Introduction, Third Edition (2009)
       ! Chap 10.4
 
       !============= Input(s) ==============!
       real(PR)   , dimension(4), intent(in) :: ul, ur
       integer              , intent(in)     :: dir
       !============= Output(s) =============!
       real(PR), dimension(4)                :: F

       Real(PR) :: vx_L, vx_R, vy_L, vy_R, p_L, p_R, c_L, c_R, a, pistar, ustar, trans_diff
       Real(PR) :: no

       logical :: adm
 
       adm = .true.
          
       call c2p_local(uL, vx_L, vy_L, no, no, p_L, c_L, adm)
          
       call c2p_local(uR, vx_R, vy_R, no, no, p_R, c_R, adm)

       a=1.1*max(uL(rho)*c_L,uR(rho)*c_R)
       
       if (dir==dir_x) then
         pistar = 0.5*(p_L+p_R) - 0.5*a*(vx_R-vx_L)
         ustar  = 0.5*(vx_L+vx_R) - 0.5*(p_R-p_L)/a
         trans_diff = -0.5*a*(vy_R-vy_L)
       else
         pistar = 0.5*(p_L+p_R) - 0.5*a*(vy_R-vy_L)
         ustar  = 0.5*(vy_L+vy_R) - 0.5*(p_R-p_L)/a
         trans_diff = -0.5*a*(vx_R-vx_L)
       end if

  

       if (adm) then

         if (ustar>0)then
            F = uL*ustar
          else
            F = uR*ustar
         end if

         if (dir==dir_x) then
           F(momx) = F(momx) + pistar
           F(momy) = F(momy) + trans_diff

         else
           F(momy) = F(momy) + pistar
           F(momx) = F(momx) + trans_diff
         end if

         F(ener) = F(ener) + pistar*ustar
 
       else
          F(:) = 1./(no-no)
       end if
 
    end function unsplit_flux
 
 
 end module mod_unsplit
 