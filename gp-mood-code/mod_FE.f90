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
   use mod_unsplit
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

      integer :: l, n, j, k, count_

      logical :: criterion_iter

      real(4) :: tic_pred, tac_pred, tic_corr, tac_corr

      if (first_RK_stage) then 
         count_detected_cell_a_posteriori = 0
         count_detected_cell_a_priori = 0
         count_detected_cell = 0
      end if

      values_NN(:,:,1)=zero
      values_NN(:,:,2)=one

      CellGPO   =  mord
      CellGPO_priori= mord

      if (((method==NN_GP_MOOD).or.(method==NN_GP_MOOD_CC)).and.(niter>nsteps_with_no_NN)) then
         call cpu_time(tic_pred)
         call compute_CellGPO_with_NN(Uin)
         call cpu_time(tac_pred)
         time_spent_predicting=time_spent_predicting+tac_pred-tic_pred
      end if

      DetCell   = .true.
      DetFace_x = .true.
      DetFace_y = .true.

      MOOD_finished = .false.

      call Boundary_C(Uin)

      Uout = Uin

      count_=0
      do while (MOOD_finished .eqv. .false.)

            call cpu_time(tic_corr)

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
                        elseif (numFlux == unsplit) then
                           Flux_gauss(rho:ener,j) =  unsplit_Flux(ul,ur,dir_x)
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
                        elseif (numFlux == unsplit) then
                           Flux_gauss(rho:ener,j) =  unsplit_Flux(ub,ut,dir_y)
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

            if ((method == GP_MOOD).or.(method == POL_MOOD).or.(((method==NN_GP_MOOD).or.(method==NN_GP_MOOD_CC)).and.(niter<=nsteps_with_no_NN))) then

               call DETECTION(Uin,Uout)

            else if ((method==NN_GP_MOOD).or.(method==NN_GP_MOOD_CC)) then

               call NN_DETECTION(Uin,Uout)

               if (MOOD_finished .eqv. .false.) then
                  steps_NN_produced_NAN(niter) = 1
                  count_steps_NN_produced_NAN=count_steps_NN_produced_NAN+1
               end if

            else
               MOOD_finished = .true.
            end if

            call cpu_time(tac_corr)

            if (count_==0) then
               time_spent_first_shot=time_spent_first_shot+tac_corr-tic_corr
            end if

            if (count_>=1) then
               time_spent_correcting = time_spent_correcting +tac_corr-tic_corr
            end if
            count_=count_+1
      end do
      
      criterion_iter=criterion_niter_f()

      if ((first_RK_stage).and.(write_NN_dataset).and.(criterion_iter)) then
         steps_NN_sample(niter) = 1
         call append_to_NN_datatset(Uin)
      end if

      if (problem == RT) then
         Uout(ener,:,:) = Uout(ener,:,:) - dt * g*Uout(momy,:,:)
         Uout(momy,:,:) = Uout(momy,:,:) - dt * g*Uout(rho,:,:)
      end if

      if ((integrator == SSP_RK4)) Uout = Uout - Uin

   end subroutine Forward_Euler

end module mod_FE
