module mod_subroutine_MOOD
  use constants
  use parameters
  use global_variables
  use physics
  use BC
  use GP_init
  implicit none

contains


  subroutine DETECTION(Uin,Uout)

    real(PR), dimension(4,lb:le, nb:ne ),intent(in   ) :: Uin
    real(PR), dimension(4,lb:le, nb:ne ),intent(inout) :: Uout


    logical , dimension(1:lf, 1:nf)                 :: decrease

    real(PR)                                        :: minlocal, Maxlocal, p, Xmin, Xmax, Ymin, Ymax, delta
    real(PR)                                        :: UL, UB, UT, UR, UM, Mm

    real(PR), dimension(-1:1, -1:1)                       :: D2x, D2y

    real, dimension(sz_cross) :: q


    logical                                         :: u2_criterion

    integer :: l, n, i, j, count, k, ord, ord_derr

    real(PR) :: divV, gradP, pL, pR, pT, pB, threshold1, threshold2, Ca, Mach
    !!real(PR) :: Mach


    MOOD_finished = .true.
    decrease      = .false.

    count = 0

    !! initialize DMP here
    DMP = .true.
    threshold1 = 5.0
    !threshold2 = max(dx**2, dy**2) !threshold1
    threshold2 = threshold1
    
    call Boundary_C(Uout)


    do n = 1, nf
       do l = 1, lf

          if ((DetCell(l,n)).and.(CellGPO(l,n) > 1)) then

             if ((Uout(rho,l,n) <= 0.).or.(ISNAN(Uout(rho,l,n)))) then
                decrease(l,n) = .true.
             else

                p = pressure(Uout(1:4,l,n))

                if ((p <= 0.).or.(ISNAN(p))) then
                   decrease(l,n) = .true.
                else


                   !! first shock-detector
                   !! nabla \cdot (rho U) = (rho \nabla \cdot U) + (U \cdot \nabla rho)
                   !! ==> \nabla \cdot U = [\nabla \cdot (rho U)]/rho - (U \cdot \nabla rho)/rho
                   divV = (Uin(momx,l+1,n  ) - Uin(momx,l-1,n  ))/dx &
                        + (Uin(momy,l,  n+1) - Uin(momy,l,  n-1))/dy

                   !! [nabla \cdot (rho U)]/rho
                   divV = 0.5*divV/Uin(rho,l,n)

                   !! [nabla \cdot (rho U)]/rho - (U \cdot \nabla rho)/rho from 
                   divV = divV - 0.5*(Uin(momx,l,n)*(Uin(rho,l+1,n  ) - Uin(rho,l-1,n  ))/dx &
                                     +Uin(momy,l,n)*(Uin(rho,l,  n+1) - Uin(rho,l,  n-1))/dy)&
                                     /Uin(rho, l,n)**2

                   !! define sound speed
                   Ca = 1.4*pressure(Uin(1:4,l,n))/Uin(rho,l,n)
                   !Ca = sqrt(Ca)
                   ! threshold2 = 1.e2

                   !! local Mach number
                   Mach =  (Uin(momx,l,n)**2 + Uin(momy,l,n)**2)/Uin(rho,l,n)**2
                   Mach = sqrt(Mach/Ca)
                   

                   !! second shock-detector
                   pL = pressure(Uin(1:4,l-1,n  ))
                   pR = pressure(Uin(1:4,l+1,n  ))
                   pB = pressure(Uin(1:4,l,  n-1))
                   pT = pressure(Uin(1:4,l,  n+1))
                   gradP = 0.5*(abs(pR-pL)/min(pL,pR)/dx + abs(pT-pB)/min(pB,pT)/dy)

                   !print*,'at cell',l,n,' divV =',divV


                   !if (divV < -threshold2*Ca/dx*0.5 .and. gradP > threshold1) then

                  
                   !! This is the *OLD* CSD check
                   if (divV < -threshold2 .and. gradP > threshold1) then
                      !print*, Mach
                      DMP = .true.
                   else
                      DMP = .false.
                   endif
                

!!$                   print*,'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
!!$                   print*,'l,n=',l,n
!!$                   print*,'divV =', divV
!!$                   print*,'Mach =', Mach
!!$                   print*,'gradP=', gradP
!!$                   print*,'==========================================='
!!$
                   !! DL -- 8/14/22
                   !! It's very weird that in 1D shock tube (Sod, Shu-Osher), even though
                   !! they are solved in 2D in x-direction, DMP order reduction never happens.
                   !! Shu-Osher looks "ok" without reduction, but Sod looks a bit oscillatory;
                   !! We've been lucky that we thought it was reducing orders until now;
                   !! In 1D shock tubes, we are using \ell = 6\Delta;
                   !! At this point, we will leave more investigation for a near future task
                   !! and wrap up the study and submit a paper.
                   !!
                   !! DL -- 8/15/22
                   !! Well, Sod 1D in a 256x256 2D domain does drop the order from 3 to 1
                   !! exactly at the shock. This means that CDS works in the new form
                   !! but not sufficient to suppress the oscillations at the tail of
                   !! the rarefaction and at the post shock regions.
                  
!!$                   if (divV < -(max(dx,dy))**2 .and. Mach > 0.2 .and. gradP > threshold1) then
!!$                      !if (divV < 0. .and. Mach > 0.2 ) then !.and. gradP > threshold1) then
!!$!$                      print*,'We are in'
!!$!$                      stop
!!$                      DMP = .true.
!!$                   else
!!$                      DMP = .false.
!!$                   endif


                   if ( (DMP) ) then

                      UB = Uin(rho,l,n-1)
                      UL = Uin(rho,l-1,n)
                      UM = Uin(rho,l,n  )
                      UR = Uin(rho,l+1,n)
                      UT = Uin(rho,l,n+1)

                      if ( dim == 1 ) then
                         UT = UM
                         UB = UM
                      end if

                      minlocal = min(UM, UL, UB, UT, UR)
                      Maxlocal = max(UM, UL, UB, UT, UR)

                      Mm = Maxlocal-Minlocal

                      !! Plateau detection
                      if ( Mm > dx**3 ) then

                         if ((Uout(rho,l,n) > Maxlocal ).or.(Uout(rho,l,n) < minlocal)) then
                            
                            decrease(l,n) = .true.

                            if (U2) then


                               do j = -1, 1
                                  do i = -1, 1

                                     if ((i == 0).or.(j==0)) then

                                        if (space_method == POL_MOOD) then

                                           !  D2x(i,j)  = dot_product(D2O4(:),Uin(rho, l+i-2:l+i+2, n+j))/dx**2
                                           !  D2y(i,j)  = dot_product(D2O4(:),Uin(rho, l+i, n+j-2:n+j+2))/dy**2

                                           D2x(i,j)  = dot_product(D2O2(:),Uin(rho, l+i-1:l+i+1, n+j))/dx**2
                                           D2y(i,j)  = dot_product(D2O2(:),Uin(rho, l+i, n+j-1:n+j+1))/dy**2

                                        else ! GP MOOD

                                           ord_derr = 3
                                           if ( cross_stencil ) then

                                              do k = 1, stcl_sz(ord_derr)
                                                 q(k) = Uin(rho,l+ixiy(ord_derr,k,1)+i, n+ixiy(ord_derr,k,2)+j)
                                              end do

                                              D2x(i,j) = dot_product(GP_d2x(ord_derr,1:stcl_sz(ord_derr)),q(1:stcl_sz(ord_derr)))
                                              D2y(i,j) = dot_product(GP_d2y(ord_derr,1:stcl_sz(ord_derr)),q(1:stcl_sz(ord_derr)))
                                           else

                                              do k = 1, stcl_sz(ord_derr)
                                                 q(k) = Uin(rho,l+ixiy_sp(ord_derr,k,1)+i, n+ixiy_sp(ord_derr,k,2)+j)
                                              end do

                                              D2x(i,j) = dot_product(GP_d2x_sp(ord_derr,1:stcl_sz(ord_derr)),q(1:stcl_sz(ord_derr)))
                                              D2y(i,j) = dot_product(GP_d2y_sp(ord_derr,1:stcl_sz(ord_derr)),q(1:stcl_sz(ord_derr)))

                                           end if



                                        end if

                                     end if


                                  end do
                               end do


                               Xmin = min(D2x(-1,0),D2x(0,0), D2x(1,0),D2x(0,-1),D2x(0,1))
                               Xmax = max(D2x(-1,0),D2x(0,0), D2x(1,0),D2x(0,-1),D2x(0,1))

                               Ymin = min(D2y(-1,0),D2y(0,0), D2y(1,0),D2y(0,-1),D2y(0,1))
                               Ymax = max(D2y(-1,0),D2y(0,0), D2y(1,0),D2y(0,-1),D2y(0,1))

                               delta = 0.0

                               if ( dim == 1 ) then
                                  if (u2_tol) delta = dx
                               else if (dim == 2) then
                                  if (u2_tol) delta = max(dx,dy)
                               end if

                               if ( (Xmin*Xmax >-delta) .and. ( (max(abs(Xmin),abs(Xmax)) < delta).or.(abs(Xmin/Xmax) >= 1./2) ) ) then
                                  decrease(l,n) = .false.
                               else if ( (Ymin*Ymax >-delta).and.( (max(abs(Ymin),abs(Ymax)) < delta).or.(abs(Ymin/Ymax) >= 1./2) )) then
                                  if ( dim == 2 ) decrease(l,n) = .false.
                               end if

                            end if

                         end if

                      end if

                   end if
                   !endif

                end if
             end if

          end if



          if (decrease(l,n) .eqv. .false.) then

             DetCell(l,n)       = .false.
             DetFace_y(l  ,n-1) = .false.
             DetFace_y(l  ,n  ) = .false.
             DetFace_x(l-1,n  ) = .false.
             DetFace_x(l  ,n  ) = .false.

          end if

       end do
    end do


    do n = 1, nf
       do l = 1, lf

          if (decrease(l,n)) then

             count = count + 1
             !  print*,l,n

             MOOD_finished = .false.

             if (CellGPO(l,n) == 3)  CellGPO(l,n) = 1
             if (CellGPO(l,n) >  3)  CellGPO(l,n) = 3

             DetCell(l  ,n-1) = .true.
             DetCell(l-1,n  ) = .true.
             DetCell(l  ,n  ) = .true.
             DetCell(l+1,n  ) = .true.
             DetCell(l  ,n+1) = .true.

             DetFace_y(l  ,n-1) = .true.
             DetFace_y(l  ,n  ) = .true.
             DetFace_x(l  ,n  ) = .true.
             DetFace_x(l-1,n  ) = .true.

             ! DL -- solution vector: rho, momx, momy, ener
             Uout(1:4,l  ,n-1) = Uin(1:4,l  ,n-1)
             Uout(1:4,l-1,n  ) = Uin(1:4,l-1,n  )
             Uout(1:4,l  ,n  ) = Uin(1:4,l  ,n  )
             Uout(1:4,l+1,n  ) = Uin(1:4,l+1,n  )
             Uout(1:4,l  ,n+1) = Uin(1:4,l  ,n+1)

          end IF

       end do
    end do

    count_FE = count_FE + count



  end subroutine DETECTION

end module mod_subroutine_MOOD
