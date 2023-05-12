module BC
   use constants
   use parameters
   use global_variables
   use IC

   implicit none

contains
   subroutine Boundary_C(U)

      real(PR), dimension(4,lb:le, nb:ne), intent(inout) :: U

      integer                                            :: l, n, k
      real(PR) :: x, y, xmin
      real(PR) , dimension(4) :: Qin, Qout

      if (BC_type ==  Neumann) then

         do n = 1-ngc, nf+ngc
            ! Right
            do l = lf+1, lf+ngc
               U(:,l,n) = U(:,lf,n)
            end do

            ! Left
            do l = 1-ngc,0
               U(:,l,n) = U(:,1,n)
            end do

         end do

         do l = 1-ngc, lf+ngc
            ! top
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf)
            end do

            ! bottom
            do n = 1-ngc,0
               U(:,l,n) = U(:,l,1)
            end do
         end do

      else  if (BC_type ==  Periodic) then

         do n = 1-ngc, nf+ngc
            ! Right
            U(:,lf+1:lf+ngc,n) = U(:,1:ngc,n)
            ! Left
            U(:,1-ngc:0,n) = U(:,lf-ngc+1:lf,n)
         end do

         do l = 1-ngc, lf+ngc
            ! Top
            U(:,l,nf+1:nf+ngc) = U(:,l,1:ngc)
            ! bottom
            U(:,l,1-ngc:0) = U(:,l,nf-ngc+1:nf)
         end do

      else if (BC_type ==  Dirichlet) then

         do n = 1-ngc, nf+ngc
            ! Right

            ! Right
            do l = lf+1, lf+ngc
               U(:,l,n) = primitive_to_conservative((/1. + 0.2*sin(5*(mesh_x(l)-4.5_pr)),0.,0.,1./))
            end do

            ! Left
            do l = 1-ngc,0
               U(:,l,n) = primitive_to_conservative((/3.857143_pr,2.629369_pr,0.,10.33333_pr/))
            end do
         end do

         do l = 1-ngc, lf+ngc
            ! top
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf)
            end do

            ! bottom
            do n = 1-ngc,0
               U(:,l,n) = U(:,l,1)
            end do
         end do

      else if (BC_type ==  Reflective) then

         do n = 1-ngc, nf+ngc
            ! Right
            k = 0
            do l = lf+1, lf+ngc

               U(:,l,n) = U(:,lf-k,n)

               k = k + 1
               U(momx,l,n) = -U(momx,l,n)
            end do

            ! Left
            k = 0
            do l = 0,1-ngc,-1
               U(:,l,n) = U(:,1+k,n)
               !U(:,l,n) = U(:,1,n)

               k = k + 1
               U(momx,l,n) = -U(momx,l,n)

            end do
         end do

         do l = 1-ngc, lf+ngc
            !top
            k = 0
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf-k)
               !U(:,l,n) = U(:,l,nf)

               k = k + 1
               U(momy,l,n) = -U(momy,l,n)

            end do

            ! bottom
            k = 0
            do n = 0, 1-ngc,-1
               U(:,l,n) = U(:,l,1+k)
               !U(:,l,n) = U(:,l,1)

               k = k + 1
               U(momy,l,n) = -U(momy,l,n)
            end do
         end do

      else if (BC_type == Mach800_BC) then
         ! [0,1]x[0,1] domain with the strong inflow BC where
         ! 0.45 <= x <= 0.55 @y=0, where the jet density = 1.4, jet velocity = 800

         !right, top, left = outflow
         do n = 1-ngc, nf+ngc
            ! Right
            do l = lf+1, lf+ngc
               U(:,l,n) = U(:,lf,n)
            end do

            ! Left
            do l = 1-ngc,0
               U(:,l,n) = U(:,1,n)
            end do

         end do

         do l = 1-ngc, lf+ngc
            ! top
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf)
            end do

         end do

         ! bottom = user
         ! bottom -- do outflow first and fix later
         do l = 1-ngc, lf+ngc
            do n = 1-ngc,0
               ! outflow elsewhere
               U(:,l,n) = U(:,l,1)
            enddo
         enddo

         ! bottom -- fix here
         do l = 1-ngc, lf+ngc
            do n = 1-ngc,0
               if (abs(mesh_x(l) - 0.75) .le. 0.05) then ! this for [0, 1.5]x[0, 1.5]
                  U(:,l,n) = primitive_to_conservative((/1.4_PR , 0.0_PR, 800.0_PR, 1.0_PR/))
               endif
            end do
         end do

      else if (BC_type == DoubleMach800_BC) then
         ! [0,1]x[0,1] domain with the strong inflow BC where
         ! 0.45 <= x <= 0.55 @y=0, where the jet density = 1.4, jet velocity = 800

         !right, top, left = outflow
         do n = 1-ngc, nf+ngc
            ! Right
            do l = lf+1, lf+ngc
               U(:,l,n) = U(:,lf,n)
            end do

            ! Left
            do l = 1-ngc,0
               U(:,l,n) = U(:,1,n)
            end do

         end do

         ! ==========================================
         ! top = user
         ! top -- do outflow first and fix later
         ! ==========================================

         do l = 1-ngc, lf+ngc
            ! top
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf)
            end do
         end do
         ! top -- fix here
         do l = 1-ngc, lf+ngc
            do n = nf+1, nf+ngc
               if (abs(mesh_x(l) - 0.75) .le. 0.05) then
                  U(:,l,n) = primitive_to_conservative((/1.4_PR , 0.0_PR, -800.0_PR, 1.0_PR/))
               endif
            end do
         end do

         ! ==========================================
         ! bottom = user
         ! bottom -- do outflow first and fix later
         ! ==========================================
         do l = 1-ngc, lf+ngc
            do n = 1-ngc,0
               ! outflow elsewhere
               U(:,l,n) = U(:,l,1)
            enddo
         enddo

         ! bottom -- fix here
         do l = 1-ngc, lf+ngc
            do n = 1-ngc,0
               if (abs(mesh_x(l) - 0.75) .le. 0.05) then
                  U(:,l,n) = primitive_to_conservative((/1.4_PR , 0.0_PR, 800.0_PR, 1.0_PR/))
               endif
            end do
         end do

      else if (BC_type == RMI_BC) then
         ! [0,6]x[0,1] domain with the strong inflow BC;
         ! reflective on top and bottom; outflow on right

         ! ==========================================
         ! Left -- inflow
         do n = 1-ngc, nf+ngc
            do l = 1-ngc, 0
               U(:,l,n) = primitive_to_conservative((/2.666666666666667, 2.*sqrt(1.4), 0., 4.5/))
            enddo
         end do

         ! ==========================================
         ! Right -- outflow
         do n = 1-ngc, nf+ngc
            do l = lf+1, lf+ngc
               U(:,l,n) = U(:,lf,n)
            end do
         end do

         ! ==========================================
         ! Top and bottom -- reflective
         do l = 1-ngc, lf+ngc
            ! top
            k = 0
            do n = nf+1, nf+ngc
               U(:,l,n) = U(:,l,nf-k)
               !U(:,l,n) = U(:,l,nf)
               k = k + 1
               U(momy,l,n) = -U(momy,l,n)
            end do

            ! bottom
            k = 0
            do n = 0, 1-ngc,-1
               U(:,l,n) = U(:,l,1+k)
               !U(:,l,n) = U(:,l,1)
               k = k + 1
               U(momy,l,n) = -U(momy,l,n)
            end do
         end do



      else if (BC_type == DMR) then

         do n = 1-ngc, nf+ngc
            ! Right Dirichlet
            do l = lf+1, lf+ngc

               U(:,l,n) = f_DMR(mesh_x(l),mesh_y(n))
               !print*,l,n,'R'
            end do

            ! Left dirichlet
            do l = 1-ngc,0
               U(:,l,n) = f_DMR(mesh_x(l),mesh_y(n))
               !print*,l,n,'L'
            end do

         end do

         do l = 1-ngc, lf+ngc
            ! top

            xmin = 1./6 + 10.*t_rk/(0.5*sqrt(3.)) + 1./sqrt(3.)

            do n = nf+1, nf+ngc
               x = mesh_x(l)
               y = mesh_y(n)
               if( x<xmin) then

                  !still in shock, use left state
                  U(:,l,n) = primitive_to_conservative((/8. , 7.1447096 , -4.125  , 116.5/))
                  !print*,l,n,'T'
               else
                  !outside of shock
                  U(:,l,n) = primitive_to_conservative((/1.4_PR , 0.0_PR, 0.0_PR  , 1.0_PR/))
                  !print*,l,n,'T'
               end if


            end do

            ! bottom

            if (mesh_x(l) > 1./6) then
               k = 0
               do n = 0, 1-ngc,-1
                  U(:,l,n) = U(:,l,1+k)
                  k = k + 1
                  U(momy,l,n) = -U(momy,l,n)
                  !  print*,l,n,'B'
               end do
            else
               do n = 0, 1-ngc,-1
                  U(:,l,n) = primitive_to_conservative((/8._PR , 7.1447096_PR , -4.125_PR  , 116.5_PR/))
                  !print*,l,n,'B'
               end do

            end if

         end do
     
         else if (BC_type == RT_BC) then 
            do n = 1-ngc, nf+ngc
               ! Right
               k = 0
               do l = lf+1, lf+ngc
   
                  U(:,l,n) = U(:,lf-k,n)
   
                  k = k + 1
                  U(momx,l,n) = -U(momx,l,n)
               end do
   
               ! Left
               k = 0
               do l = 0,1-ngc,-1
                  U(:,l,n) = U(:,1+k,n)
                  !U(:,l,n) = U(:,1,n)
   
                  k = k + 1
                  U(momx,l,n) = -U(momx,l,n)
               end do
            end do

            do l = 1-ngc, lf+ngc
               !top
               k = 0
               do n = nf+1, nf+ngc
                  Qin(:) = conservative_to_primitive(U(:,l,nf-k))
                  Qout(rho)  =  Qin(rho)
                  Qout(momx) =  Qin(momx)
                  Qout(momy) = -Qin(momy)
                  Qout(ener) = Qin(ener) - 2*k*dy*g*Qin(rho)
                  U(:,l,n)= primitive_to_conservative(Qout)
                  k = k + 1
               end do
   
               ! bottom
               k = 0
               do n = 0, 1-ngc,-1
                  Qin(:) = conservative_to_primitive(U(:,l,1+k))
                  Qout(rho)  =  Qin(rho)
                  Qout(momx) =  Qin(momx)
                  Qout(momy) = -Qin(momy)
                  Qout(ener) = Qin(ener) + 2*k*dy*g*Qin(rho)
                  U(:,l,n)= primitive_to_conservative(Qout)
                  k = k + 1
               end do
            end do

         else

         print*, 'BC type not programmed'
      end if


   end subroutine Boundary_C

end module BC
