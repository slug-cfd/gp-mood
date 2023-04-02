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
  use mod_write_NN_dataset
  use mod_NN

  implicit none

contains

  subroutine Forward_Euler(Uin, Uout, first)

    real(PR), intent(inout)    , dimension(4,lb:le, nb:ne) :: Uin
    real(PR), intent(out)      , dimension(4,lb:le, nb:ne) :: Uout

    logical, intent(in) :: first

    real(PR), dimension(4)                                 :: ul, ur, ut, ub
    real(PR), dimension(4,ngp)                             :: Flux_gauss

    real(PR)   , dimension(4, -1:lf+1,  0:nf+1  ) :: L_Flux_x
    real(PR)   , dimension(4,  0:lf+1  , -1:nf+1) :: L_Flux_y

    integer :: l, n, j, k, NWRONG, NSMOOTH, NSHARP

    logical :: criterion_iter

    if (space_method .ne. eval_NN_GP_MOOD) then
      count_detected_cell = 0
      count_NN_PAD = 0
      CellGPO   =  Mord
      if (space_method == NN_GP_MOOD) then 
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

         if ((space_method == GP_MOOD).or.(space_method == POL_MOOD)) then

            call DETECTION(Uin,Uout)

         else if (space_method == NN_GP_MOOD) then 

            call NN_DETECTION(Uin,Uout)

            if (MOOD_finished .eqv. .false.) then 
               count_NN_need_posteriori_correction=count_NN_need_posteriori_correction+1
            end if

         else
            MOOD_finished = .true.
         end if

      end do
    else
      !Compute MOOD order
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

         call DETECTION(Uin,Uout)

      end do
      !Save the order
      CellGPO_MOOD=CellGPO
      !Reset and compute solution with NN_GP_MOOD
      count_detected_cell = 0
      count_NN_PAD = 0
      CellGPO   =  Mord
      call compute_CellGPO_with_NN(Uin)
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

         call NN_DETECTION(Uin,Uout)
         
         if (MOOD_finished .eqv. .false.) then 
            count_NN_need_posteriori_correction=count_NN_need_posteriori_correction+1
         end if
         
      end do


      NWRONG=0
      NSMOOTH=0
      NSHARP=0
      do n = 1, nf
         do l = 1, lf
            if (CellGPO(l,n)>CellGPO_MOOD(l,n)) then 
               NSHARP=NSHARP+1
               NWRONG=NWRONG+1
            else if  (CellGPO(l,n)<CellGPO_MOOD(l,n)) then 
               NSMOOTH=NSMOOTH+1
               NWRONG=NWRONG+1
            end if
         end do 
      end do

     print*, NWRONG*100.0/(lf*nf), NSMOOTH*100.0/NWRONG, NSHARP*100.0/NWRONG

    end if


    if ((first).and.(write_NN_dataset).and.(criterion_iter)) then
      criterion_iter=criterion_niter_f()
      call write_NN_datatset(Uin, CellGPO)
    end if

    if ((time_method == SSP_RK4)) Uout = Uout - Uin

  end subroutine Forward_Euler

end module mod_FE
