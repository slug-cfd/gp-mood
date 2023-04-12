module mod_FE
   use constants
   use parameters
   use global_variables
   use reconstruction
   use BC
   use mod_subroutine_MOOD
   use mod_subroutine_NN_MOOD
   use mod_HLLC
   use mod_LLF
   use mod_HLL
   use mod_append_NN_dataset
   use mod_NN

   implicit none

contains

   subroutine Forward_Euler(Uin, Uout, first_RK_stage)

      real(PR), intent(inout)    , dimension(4,lb:le, nb:ne) :: Uin
      real(PR), intent(out)      , dimension(4,lb:le, nb:ne) :: Uout

      logical, intent(in) :: first_RK_stage

      real(PR), dimension(4)                                 :: ul, ur, ut, ub
      real(PR), dimension(4,ngp)                             :: Flux_gauss

      real(PR)   , dimension(4, -1:lf+1,  0:nf+1  ) :: L_Flux_x
      real(PR)   , dimension(4,  0:lf+1  , -1:nf+1) :: L_Flux_y

      integer :: l, n, j, k

      logical :: criterion_iter

      if (first_RK_stage) then 
         count_detected_cell_RK = 0
         count_NN_PAD_RK = 0
      end if

      CellGPO   =  Mord

      if (method == NN_GP_MOOD) then
         call compute_CellGPO_with_NN(Uin)
      end if

      DetCell   = .true.
      DetFace_x = .true.
      DetFace_y = .true.

      MOOD_finished = .false.

      call Boundary_C(Uin)

      Uout = Uin

      do while (MOOD_finished .eqv. .false.)

            call Recons(Uin)

            do n = 0, nf
               do l = 0, lf

                  if (DetFace_x(l,n)) then

                     do j = 1, ngp
                        uL =  Uh(: ,iR,j,l  ,n)
                        uR =  Uh(: ,iL,j,l+1,n)

                        if (numFlux == HLLC) then
                           Flux_gauss(rho:ener,j) =  HLLC_Flux(ul,ur,dir_x)
                        elseif (numFlux == LLF) then
                           Flux_gauss(rho:ener,j) =  LLF_Flux(ul,ur,dir_x)
                        elseif (numFlux == HLL) then
                           Flux_gauss(rho:ener,j) =  HLL_Flux(ul,ur,dir_x)
                        endif

                     end do

                     do k = rho, ener
                        L_Flux_x(k,l,n) = dot_product(Flux_gauss(k,1:ngp),gauss_weight(ngp,1:ngp))
                     end do

                     Uout(:,l  ,n) = Uout(:,l  ,n) - dt/dx * L_Flux_x(:,l,n)
                     Uout(:,l+1,n) = Uout(:,l+1,n) + dt/dx * L_Flux_x(:,l,n)
                  else

                     if (DetCell(l  ,n)) Uout(:,l  ,n) = Uout(:,l  ,n) - dt/dx * L_Flux_x(:,l,n)
                     if (DetCell(l+1,n)) Uout(:,l+1,n) = Uout(:,l+1,n) + dt/dx * L_Flux_x(:,l,n)
                  end if


                  if (DetFace_y(l,n)) then


                     do j = 1, ngp
                        uB =  Uh(: ,iT,j,l,n  )
                        uT =  Uh(: ,iB,j,l,n+1)

                        if (numFlux == HLLC) then
                           Flux_gauss(rho:ener,j) =  HLLC_Flux(ub,ut,dir_y)
                        elseif (numFlux == LLF) then
                           Flux_gauss(rho:ener,j) =  LLF_Flux(ub,ut,dir_y)
                        elseif (numFlux == HLL) then
                           Flux_gauss(rho:ener,j) =  HLL_Flux(ub,ut,dir_y)
                        endif

                     end do

                     do k = rho, ener
                        L_Flux_y(k,l,n) = dot_product(Flux_gauss(k,1:ngp),gauss_weight(ngp,1:ngp))
                     end do



                     Uout(:,l,n  ) = Uout(:,l,n  ) - dt/dy * L_Flux_y(:,l,n)
                     Uout(:,l,n+1) = Uout(:,l,n+1) + dt/dy * L_Flux_y(:,l,n)
                  else


                     if (DetCell(l,n  )) Uout(:,l,n)   =  Uout(:,l,n  ) - dt/dy * L_Flux_y(:,l,n)
                     if (DetCell(l,n+1)) Uout(:,l,n+1) =  Uout(:,l,n+1) + dt/dy * L_Flux_y(:,l,n)

                  end if

               end do
            end do

            if ((method == GP_MOOD).or.(method == POL_MOOD)) then

               call DETECTION(Uin,Uout)

            else if (method == NN_GP_MOOD) then

               call NN_DETECTION(Uin,Uout)

               if (MOOD_finished .eqv. .false.) then
                  steps_NN_produced_NAN(niter) = 1
                  count_steps_NN_produced_NAN=count_steps_NN_produced_NAN+1
               end if

            else
               MOOD_finished = .true.
            end if

      end do
      
      criterion_iter=criterion_niter_f()

      if ((first_RK_stage).and.(write_NN_dataset).and.(criterion_iter)) then
      !if ((write_NN_dataset).and.(criterion_iter)) then
         call append_to_NN_datatset(Uin)
      end if

      if ((integrator == SSP_RK4)) Uout = Uout - Uin

   end subroutine Forward_Euler

end module mod_FE
