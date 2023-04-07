module mod_POL_MOOD
  use constants
  use parameters
  use global_variables
  use physics

  implicit none

contains

  subroutine POL_MOOD_(Uin)

    real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

    integer                     :: l, n, k, i,j
    real(PR), dimension(4,sz_cross) :: q
    logical                     :: reconstruction, done
    integer                     :: ord

    real, dimension(4) :: u
    real(PR), dimension(4,5) :: coef_mpt
    real(PR), dimension(4,3) :: coef_mpt1d

    logical :: multiD_polyRecon

    multiD_polyRecon = .false.


    ! define the poly3 coeffs for the mid-point rule
    ! the first index goes 1 through 4, according to the order iL=1, iT=2, iR=3, iB=4
    coef_mpt(1,:) = (/11./12.,  1./3.,  -1./24., -1./6.,  -1./24. /) !for iL
    coef_mpt(2,:) = (/11./12., -1./24.,  1./3.,  -1./24., -1./6.  /) !for iT
    coef_mpt(3,:) = (/11./12., -1./6.,  -1./24.,  1./3.,  -1./24. /) !for iR
    coef_mpt(4,:) = (/11./12., -1./24., -1./6.,  -1./24.,  1./3.  /) !for iB



    coef_mpt1d(1,:) = (/ 1./3.,  5./6., -1./6. /) !for iL
    coef_mpt1d(2,:) = (/-1./6.,  5./6.,  1./3. /) !for iT
    coef_mpt1d(3,:) = (/-1./6.,  5./6.,  1./3. /) !for iR
    coef_mpt1d(4,:) = (/ 1./3.,  5./6., -1./6. /) !for iB
    
 
    do n = 0, nf+1
       do l = 0, lf+1

          q = 0.
          done = .false.

          ! iL = 1, iT = 2, iR = 3, iB = 4 (see constants.f90)
          do i = iL,iB

             reconstruction = .false.
!print*,'(n,l)=',n,l
             if ((i==iL).and.(DetFace_x(l-1,n))) then
                ord = min( CellGPO(l-1,n  ), CellGPO(l,n))
!print*,'iL',CellGPO(l-1,n  ), CellGPO(l,n),ord
                reconstruction = .true.
             else if ((i==iR).and.(DetFace_x(l,n))) then
                ord = min( CellGPO(l+1,n  ), CellGPO(l,n))
!print*,'iR', CellGPO(l+1,n  ), CellGPO(l,n), ord       
                reconstruction = .true.
             else if ((i==iB).and.(DetFace_y(l,n-1))) then
                ord = min( CellGPO(l,n -1), CellGPO(l,n))
!print*,'iB',  CellGPO(l,n -1), CellGPO(l,n), ord
                reconstruction = .true.
             else if ((i==iT).and.(DetFace_y(l,n))) then
                ord = min( CellGPO(l,n+1  ), CellGPO(l,n))
!print*,'iT', CellGPO(l,n+1  ), CellGPO(l,n), ord  
                reconstruction = .true.
             end if


             if (reconstruction .eqv. .true.) then

                if (done .eqv. .false.) then
                   do j = 1, sz_cross
                      q(:,j) = Uin(:,l+ixiy(mord,j,1),n+ixiy(mord,j,2))
                   end do

                   done = .true.

                end if


                do j = 1, ngp

                   if (ngp == 1) then
                      if (ord == 3) then
                         !print*, 'This pol mood is not programmed'
                         !stop
                         ! now supports 1-pt quadrature rule (aka, the mid-point rule)
                         ! i.e., ngp = 1 (so j = 1 always)
                         do k = rho, ener

                            if (multiD_polyRecon) then
                               ! multiD reconstruction -- toggle on and off above
                               Uh(k,i,j,l,n) = dot_product(coef_mpt(i,:),q(k,1:2*ord-1)) !multiD poly

                            else
                               ! 1D reconstruction 
                               if (i == iL .or. i == iR) then
                                  ! x-direction (l)
                                  Uh(k,i,j,l,n) = dot_product(coef_mpt1d(i,:),Uin(k,l-1:l+1,n)) !1D poly
                               elseif (i == iB .or. i == iT) then
                                  ! y-direction (n)
                                  Uh(k,i,j,l,n) = dot_product(coef_mpt1d(i,:),Uin(k,l,n-1:n+1)) !1D poly
                               endif
                            endif
                         enddo

                      elseif (ord == 1) then
                         ! this is the FOG only mode   
                         Uh(:,i,j,l,n) = Uin(:,l,n)

                         !if ((ord == 5).and.(ngp /= 2)) then
                      else if (ord == 5) then
                         print*, 'This pol mood is not programmed'
                         stop
                      else
                         ! this is the FOG only mode   
                         Uh(:,i,j,l,n) = Uin(:,l,n)
                      end if

                   else !if ngp > 1
                      if (ord == 5) then

                         do k = rho, ener
                            Uh(k,i,j,l,n) = dot_product( Pol_zT_o5(1:2*ord-1,i,j),q(k,1:2*ord-1) )
                         end do

                      else if (ord == 3) then

                         do k = rho, ener
                            Uh(k,i,j,l,n) = dot_product( Pol_zT_o3(1:2*ord-1,i,j),q(k,1:2*ord-1) )
                         end do

                      else
                         ! fog
                         Uh(:,i,j,l,n) = Uin(:,l,n)
                      end if
                   endif ! if (ngp == 1) then

                end do ! do j=1,ngp
             endif ! if (reconstruction .eqv. .true.) then

             !end if

          end do
       end do

    end do



  end subroutine POL_MOOD_

end module mod_POL_MOOD
