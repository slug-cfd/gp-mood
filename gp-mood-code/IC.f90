module IC
   use parameters
   use global_variables
   use physics
   !use random
   implicit none

contains

   subroutine InitialCdt()
      ! Uses the global variables
      !  mesh_x, mesh_y
      !
      ! Acts on the global variables
      ! U
      !
      ! Local variables:

      integer  :: l, n, seed
      real(PR) :: r,int,xx,yy, r1,r2, x_p
      real(PR) :: slope_dens, dens_amb ! for mach 800 with varying ambient density
      !real(PR) :: densR, densL, presR, presL, velxR, velxL, velyR, velyL
      real(PR) :: sim_posn, sim_shockPosn, sim_xcos, sim_ycos, sim_xangle
      ! Initialize the random number seed
      call random_seed()

      !Instructions

      do n = nb, nf
         do l = lb, lf

            if (problem == Sodx) then ! Lx = Ly= 1 = 0.2
               if (mesh_x(l) <  0.5) U(:,l,n) = primitive_to_conservative((/1.,0.,0.,1./))
               if (mesh_x(l) >= 0.5) U(:,l,n) = primitive_to_conservative((/0.125,0.,0.,0.1/))

            elseif (problem == SlowShock) then ! LeVeque page 348
               if (mesh_x(l) <  15.) U(:,l,n) = primitive_to_conservative((/5.6698, -1.4701, 0.0, 100.0/))
!!$             if (mesh_x(l) >= 15.) U(:,l,n) = primitive_to_conservative((/1.0,    -10.5,   0.0, 1.0  /))
               if (mesh_x(l) >= 15.) U(:,l,n) = primitive_to_conservative((/1.0,    -2.315811,   0.0, 1.0  /))

            else if (problem == Sod_rotated) then ! Lx = Ly= 2*sqrt(2) tmax = 0.2

               x_p = mesh_x(l)*cos(Pi/4) + mesh_y(n)*sin(Pi/4)

               if ( (x_p< 0.5).or.((1.5<x_p).and.(x_p<=2.5)).or.((3.5<x_p).and.(x_p<4)) ) then
                  U(:,l,n) = primitive_to_conservative((/1.,0.,0.,1./))
               else
                  U(:,l,n) = primitive_to_conservative((/0.125,0.,0.,0.1/))
               end if

            else if (problem == Sody) then ! Lx = Ly= 1 = 0.2
               if (mesh_y(n) <  0.5) U(:,l,n) = primitive_to_conservative((/1.,0.,0.,1./))
               if (mesh_y(n) >= 0.5) U(:,l,n) = primitive_to_conservative((/0.125,0.,0.,0.1/))


            else if (problem == Lax) then ! Lx = 2, t = 0.26
               if (mesh_x(l) <  1.) U(:,l,n) = primitive_to_conservative((/0.445, 0.698, 0.,3.528/))
               if (mesh_x(l) >= 1.) U(:,l,n) = primitive_to_conservative((/0.5  ,     0.,0. , 0.571/))

            else if (problem == Sodxy) then
               if (mesh_y(n) <   -mesh_x(l)+(sqrt(2.)-1)) U(:,l,n) = primitive_to_conservative((/1.,0.,0.,1./))
               if (mesh_y(n) >=  -mesh_x(l)+(sqrt(2.)-1)) U(:,l,n) = primitive_to_conservative((/0.125,0.,0.,0.1/))


            else if (problem == Shu_Osher) then  !Lc = 9, tmax = 1.8
               if (mesh_x(l)<= 0.5) U(:,l,n) = primitive_to_conservative((/3.857143_pr,2.629369_pr,0.,10.33333_pr/))
               if (mesh_x(l)>  0.5 ) U(:,l,n) = primitive_to_conservative((/1. + 0.2*sin(5*(mesh_x(l)-4.5_pr)),0.,0.,1./))

            else if (problem == Shu_Osher_rotated) then ! Lx = Ly= 10*2*sqrt(2) tmax = 1.8

               x_p = mesh_x(l)*cos(Pi/4) + mesh_y(n)*sin(Pi/4)

               if ( (x_p<= 1.0 ).or.((11.<x_p).and.(x_p<=21.)).or.((31.<x_p).and.(x_p<40.)) ) then

                  U(:,l,n) = primitive_to_conservative((/3.857143_pr,2.629369_pr*COS(PI/4.),2.629369_pr*SIN(PI/4.),10.33333_pr/))

               else

                  U(:,l,n) = primitive_to_conservative((/1. + 0.2*sin(5*x_p),0.,0.,1./))

               end if


            else if (problem == strong_raref) then !L=1 t = 0.15
               if (mesh_x(l)<= 0.5) U(:,l,n) = primitive_to_conservative((/1.,-2.,0.,0.4/))
               if (mesh_x(l)> 0.5) U(:,l,n) = primitive_to_conservative((/1., 2.,0.,0.4/))


            else if (problem == BLAST) then !L=1, refelxive t =0.038
               if (mesh_x(l) >= 0.)  U(:,l,n) = primitive_to_conservative((/1.,0.,0.,1000./))
               if (mesh_x(l) >= 0.1) U(:,l,n) = primitive_to_conservative((/1.,0.,0.,0.1  /))
               if (mesh_x(l) >= 0.9) U(:,l,n) = primitive_to_conservative((/1.,0.,0.,100. /))



            else if (problem == Lin_Gauss_x) then! Lx = Ly= 1


               xx = mesh_x(l)
               yy = mesh_y(n)

               int  = -0.0886227*erf(5. - 10.*(xx+dx/2)) + 0.0886227*erf(5. - 10.*(xx-dx/2)) + dx

               int  = int/dx

               print*, int, 1. + exp(-100*(xx-0.5)**2)

               U(:,l,n) = primitive_to_conservative((/int ,1.,0.,1./y/))

            else if (problem == Lin_Gauss_y) then !Lx = Ly= 1 = tmax
               r =  (mesh_y(n)-0.5)**2 + (mesh_x(l)-0.5)**2

               xx = mesh_x(l)
               yy = mesh_y(n)

               int = (-0.0886227*erf(-5*dx - 10*xx +5.) + 0.0886227*erf(5*dx - 10*xx + 5))
               int = int*(-0.0886227*erf(-5*dy - 10*yy +5.) + 0.0886227*erf(5*dy - 10*yy + 5))

               int  = (int + dx*dy)/(dx*dy)

               U(:,l,n) = primitive_to_conservative((/int ,0.,-1.,1./y/))
               !U(:,l,n) = primitive_to_conservative((/1. + exp(-100*r) ,0.,1.,1./y/))


            else if (problem == Lin_Gauss_xy) then
               r =  (mesh_x(l)-0.5)**2 + (mesh_y(n)-0.5)**2
               U(:,l,n) = primitive_to_conservative((/1. + exp(-100*r) ,1.,1.,1./y/))
               xx = mesh_x(l)
               yy = mesh_y(n)

               int = (-0.0886227*erf(-5*dx - 10*xx +5.) + 0.0886227*erf(5*dx - 10*xx + 5))
               int = int*(-0.0886227*erf(-5*dy - 10*yy +5.) + 0.0886227*erf(5*dy - 10*yy + 5))

               int  = (int + dx*dy)/(dx*dy)

               U(:,l,n) = primitive_to_conservative((/int ,1.,0.,1./y/))
               ! U(:,l,n) = primitive_to_conservative((/1. + exp(-100*r) ,1.,1.,1./y/))

            else if (problem == implosion) then !t=2.5 Lx=Ly=0.3 ! reflective
               U(:,l,n) = f_implosion(mesh_x(l),mesh_y(n))
               !if (n>=l) U(:,l,n) = f_implosion(mesh_x(n),mesh_y(l))

            else if (problem == explosion) then !t=2.5 Lx=Ly=0.3
               if (n<l ) U(:,l,n) = f_explosion(mesh_x(l),mesh_y(n))
               if (n>=l) U(:,l,n) = f_explosion(mesh_x(n),mesh_y(l))

            else if (problem == RP_2D_3) then !t =0.3 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_3(mesh_x(l),mesh_y(n))

            else if (problem == RP_2D_4) then !t =0.25 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_4(mesh_x(l),mesh_y(n))

            else if (problem == RP_2D_6) then !t =0.3 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_6(mesh_x(l),mesh_y(n))

            else if (problem == RP_2D_12) then !t =0.25 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_12(mesh_x(l),mesh_y(n))

            else if (problem == RP_2D_15) then !t =0.2 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_15(mesh_x(l),mesh_y(n))

            else if (problem == RP_2D_17) then !t =0.3 Lx = Ly = 1
               U(:,l,n) = f_RP_2D_17(mesh_x(l),mesh_y(n))

            else if (problem == DMR) then !Lx = 4, Ly = 1 tmax =0.25
               U(:,l,n) = f_DMR(mesh_x(l),mesh_y(n))

            else if (problem == sedov) then !Lx = 1, Ly = 1, tmax = 0.05
               U(:,l,n) = f_sedov(mesh_x(l),mesh_y(n))

            else if (problem == Mach800 .or. problem == DoubleMach800) then ! DL -- added the Mach 800 jet problem

               if (problem == Mach800) then
                  !! default setup
                  slope_dens = 0.
                  dens_amb   = 14.
               elseif (problem == DoubleMach800) then
                  slope_dens = (0.14 - 14.)/Ly
                  dens_amb   = slope_dens * mesh_y(n) + 14.
               endif

               U(:,l,n) = primitive_to_conservative((/dens_amb, 0.0, 0.0, 1.0/))

               ! DL -- this is a hack that seems to correct the height of the jet but I am not sure why;
               !    -- initialize the first interior cells with the jet profile
               if ((abs(mesh_x(l) - 0.5*Lx) .le. 0.05).and.(mesh_y(n) < dy)) then
                  if (problem == Mach800) then
                     U(:,l,n) = primitive_to_conservative((/1.4_PR , 0.0_PR, 800.0_PR, 1.0_PR/))
                  endif
               endif


            else if (problem == RMI) then !Lx = 6, Ly = 1, tmax = 2.0

               sim_xangle = 135.
               sim_xangle = sim_xangle * 0.017453292519943295        ! Convert to radians.
               sim_xcos = cos(sim_xangle)
               sim_ycos = sqrt(1. - sim_xcos*sim_xcos)
               sim_shockPosn = 0.8
               sim_posn = 1.0
               sim_posn = mesh_y(n) + 1.0

               U(:,l,n) = f_rmi(mesh_x(l), sim_shockPosn, sim_posn)


            else if (problem == KH) then

               r1=normal_random()*exp(-((mesh_y(n)-0.5)/(0.1))**2)
               r2=normal_random()*exp(-((mesh_y(n)-0.5)/(0.1))**2)

               if ((mesh_y(n)>=0.25).and.(mesh_y(n)<=0.75)) then
                  U(:,l,n) = primitive_to_conservative((/2.,  0.5+r1, r2, 2.5/))
               else
                  U(:,l,n) = primitive_to_conservative((/1., -0.5+r1, r2, 2.5/))
               end if

            else  if (problem == isentropic_vortex) then
               U(:,l,n) = (1./(dx*dy))*quadrature(mesh_x(l)-dx/2,mesh_x(l)+dx/2, mesh_y(n)-dy/2,mesh_y(n)+dy/2, 0.5*Lx, 0.5*Ly, 0.)

            else if (problem==RT) then 
               U(:,l,n) = f_RT(mesh_x(l),mesh_y(n))

            end if

         end do
      end do
   end subroutine InitialCdt

   function normal_random()
      implicit none
      double precision :: normal_random, u1, u2
      double precision, parameter :: pi = 3.14159265358979323846

      call random_number(u1)
      call random_number(u2)

      normal_random = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
  end function normal_random


   function f_RP_2D_3(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r
      if ((x<=0.8).and.(y<=0.8)) r = primitive_to_conservative((/0.138_PR , 1.206_PR , 1.206_PR  , 0.029_PR/))!BL
      if ((x>=0.8).and.(y<=0.8)) r = primitive_to_conservative((/0.5323_PR, 0.0_PR   , 1.206_PR  , 0.3_PR/)) !BR
      if ((x<=0.8).and.(y>=0.8)) r = primitive_to_conservative((/0.5323_PR, 1.206_PR , 0.0_PR    , 0.3_PR/)) !TL
      if ((x>=0.8).and.(y>=0.8)) r = primitive_to_conservative((/1.5_PR   , 0.0_PR   , 0.0_PR    , 1.5_PR/)) !TR
   end function f_RP_2D_3


   function f_RP_2D_4(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r
      if ((x<=0.5).and.(y<=0.5)) r = primitive_to_conservative((/1.1_PR , 0.8939_PR  , 0.8939_PR  , 1.1_PR/))!BL
      if ((x>=0.5).and.(y<=0.5)) r = primitive_to_conservative((/0.5065_PR, 0.0_PR   , 0.8939_PR  , 0.35_PR/)) !BR
      if ((x<=0.5).and.(y>=0.5)) r = primitive_to_conservative((/0.5065_PR, 0.8939_PR, 0.0_PR    , 0.35_PR/)) !TL
      if ((x>=0.5).and.(y>=0.5)) r = primitive_to_conservative((/1.1_PR   , 0.0_PR   , 0.0_PR    , 1.1_PR/)) !TR
   end function f_RP_2D_4

   function f_RP_2D_6(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      if ((x<=0.5).and.(y>=0.5)) r = primitive_to_conservative((/2.0, 0.75 , 0.5    , 1.0/)) !TL
      if ((x>=0.5).and.(y>=0.5)) r = primitive_to_conservative((/1.0   , 0.75   , -0.5    , 1.0/)) !TR
      if ((x<=0.5).and.(y<=0.5)) r = primitive_to_conservative((/1.0 , -0.75 , 0.5  , 1.0/))!BL
      if ((x>=0.5).and.(y<=0.5)) r = primitive_to_conservative((/3.0, -0.75   , -0.5  , 1.0/)) !BR

   end function f_RP_2D_6

   function f_RP_2D_12(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      if ((x<=0.5).and.(y<=0.5)) r = primitive_to_conservative((/0.8 , 0.0 , 0.0  , 1.0/))!BL
      if ((x>0.5).and.(y<=0.5)) r = primitive_to_conservative((/1.0, 0.0   ,  0.7276  , 1.0/)) !BR
      if ((x<0.5).and.(y>=0.5)) r = primitive_to_conservative((/1.0, 0.7276 , 0.0    , 1.0/)) !TL
      if ((x>=0.5).and.(y>=0.5)) r = primitive_to_conservative((/0.5313   , 0.0   , 0.0    , 0.4/)) !TR
   end function f_RP_2D_12


   function f_RP_2D_15(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      if ((x<=0.5).and.(y<=0.5)) r = primitive_to_conservative((/0.8 , 0.1 , -0.3  , 0.4/))!BL
      if ((x>=0.5).and.(y<=0.5)) r = primitive_to_conservative((/0.5313, 0.1   ,  0.4276  , 0.4/)) !BR
      if ((x<=0.5).and.(y>=0.5)) r = primitive_to_conservative((/0.5197 , -0.6259 , -0.3  , 0.4/))!TL
      if ((x>=0.5).and.(y>=0.5)) r = primitive_to_conservative((/1.0   , 0.1   , -0.3    , 1.0/)) !TR
   end function f_RP_2D_15

   function f_RP_2D_17(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      if ((x<=0.5).and.(y<=0.5)) r = primitive_to_conservative((/1.0625_PR, 0.0_PR    , 0.2145_PR  , 0.4_PR/))!BL
      if ((x>=0.5).and.(y<=0.5)) r = primitive_to_conservative((/0.5197_PR   , 0.0_PR   , -1.1259_PR   , 0.4_PR/)) !BR
      if ((x<=0.5).and.(y>=0.5)) r = primitive_to_conservative((/2.0_PR, 0.0_PR    , -0.3_PR, 1.0_PR/)) !TL
      if ((x>=0.5).and.(y>=0.5)) r = primitive_to_conservative((/1.0_PR   , 0.0_PR   , -0.4_PR    , 1.0_PR/)) !TR

   end function f_RP_2D_17

   function f_implosion(x,y)result(r)
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      real(PR) :: xx, yy
      xx = x + dx*sqrt(2.)*0.5
      yy = y + dy*sqrt(2.)*0.5

      !slow start
      ! new IC
      if (yy+xx>0.15) then
         r = primitive_to_conservative((/1. , 0. , 0.  , 1./))
      else
         r = primitive_to_conservative((/0.125 , 0. , 0.  , 0.14/))
      end if
   end function f_implosion

   function f_explosion(x,y)result(r) !Lx = Ly = 1.5, Neumann, tmax = 3.2
      real(PR), intent(in) :: x, y
      real(PR),dimension(4):: r

      real(PR) :: radius

      radius = sqrt(x**2 + y**2)

      if (radius < 0.4) then
         r = primitive_to_conservative((/1. , 0. , 0.  , 1./))
      else
         r = primitive_to_conservative((/0.125 , 0. , 0.  , 0.1/))
      end if

   end function f_explosion

   function f_DMR(x,y)result(r)

      real(PR), intent(in)  :: x, y
      real(PR),dimension(4) :: r

      real(PR) ::  xmin, ymin

      xmin = 1./6
      ymin = (x-xmin)*sqrt(3.)

      if (y > ymin) then
         !in the shocked region (left)
         r(:) = primitive_to_conservative((/8. , 7.1447096 , -4.125  , 116.5/))

      else
         !outside the shocked region (right)
         r(:) = primitive_to_conservative((/1.4 , 0.0, 0.0  , 1.0/))

      end if
   end function f_DMR


   function f_sedov(x,y)result(r)

      real(PR), intent(in)  :: x, y
      real(PR),dimension(4) :: r

      real(PR) :: E = 1.,  dr2 = (3.5*MIN(dx, dy))**2, pi = 4*atan(1.), r2

      r2= (x-0.5)**2 + (y-0.5)**2

      if (sqrt(r2) <  sqrt(dr2)) then
         r = primitive_to_conservative((/1.0, 0.0, 0.0, (1.4-1.)*E/(pi*dr2)/))
      else
!!$       r = primitive_to_conservative((/1.0, 0.0, 0.0, 1./))
         r = primitive_to_conservative((/1.0, 0.0, 0.0, 1.e-5/))
      end if
   end function f_sedov

   function f_isentropic_vortex(x,y,xc,yc,t)result(res)

      real(PR), intent(in)  :: x, y, xc, yc,t
      real(PR),dimension(4) :: res

      real :: beta = 5.0, rho, vx, vy,r, p, xp, yp, xcp, ycp

      xp = x - t
      yp = y - t

      do while (xp < 0.)
         !xp = xp +10.0
         xp = xp + 0.5*Lx
      end do

      do while (yp < 0.)
         !yp = yp +10.0
         yp = yp + 0.5*Ly
      end do

      xcp = xc
      ycp = yc

      r = sqrt( (xp-xcp)*(xp-xcp) + (yp-ycp)*(yp-ycp) )

      rho = (1.-(1.4 -1.)*(beta*beta/(8*1.4*pi*pi))*exp(1.-r*r))**(1./(1.4-1.))
      vx  = 1. - 0.5*(beta/pi) * exp(0.5*(1.-r*r))*(yp-xc)
      vy  = 1. + 0.5*(beta/pi) * exp(0.5*(1.-r*r))*(xp-yc)
      p   = rho**(1.4)

      res = primitive_to_conservative((/rho, vx, vy, p/))

   end function f_isentropic_vortex


   function f_rmi(x,shockLoc,posn) result(r)

      real(PR), intent(in) :: x, shockLoc, posn
      real(PR), dimension(4) :: r
      real(PR) :: densR, densL, presR, presL, velxR, velxL, velyR, velyL
      real(PR) :: dens, velx, vely, pres
      real(PR) :: gam,gamp,gamm,mach2,shockMach

      gam  = 1.4
      gamp = gam + 1.
      gamm = gam - 1.
      shockMach = 2.
      mach2 = shockMach*shockMach

      densL = 1.
      presL = 1.
      velxL = 0.
      velyL = 0.

      densR = 3.
      presR = 1.
      velxR = 0.
      velyR = 0.

!!$    print*,shockLoc, posn


      !! First initialize the post-shock values
      if (x .lt. shockLoc) then
         dens = densL*(gamp*mach2)/(gamm*mach2 + 2.0) !=2.66667
         pres = presL*(2.0*gam*mach2-gamm)/gamp       !=4.5
         velx = shockMach*sqrt(gam*presL/densL)       !=2*sqrt(1.4)=2.3664320
         vely = velyL
!!$       print*,'x<0.8'
!!$       print*,x
!!$       dens = 2.66667
!!$       pres = 4.5       !=4.5
!!$       velx = 2*sqrt(1.4) !=2.3664320
!!$       vely = 0.

         !print*,dens-2.66667, pres-4.5, velx-2.3664320,vely
      elseif (x .ge. shockLoc .and. x .lt. posn) then
!!$       !print*,x,y,shockLoc,posn
         dens = densL
         pres = presL
         velx = velxL
         vely = velyL
      else
!!$       print*,'x>0.8'
!!$       print*,x
         dens = densR
         pres = presR
         velx = velxR
         vely = velyR
      endif
      r =  primitive_to_conservative((/dens, velx, vely, pres/))

   end function f_rmi

   function f_RT(x,y)result(res)
      real(PR), intent(in)  :: x, y
      real(PR),dimension(4) :: res

      real(PR) :: dens, pres, vy 

      real(4) :: density_bottom=1, density_top=2, pres_bottom=5.0/2, pert=0.01

      if (y<Ly/2) then 
         dens=density_bottom
      else 
         dens=density_top
      end if

      pres=pres_bottom+g*y

      vy = pert * (1+cos(2*pi*(x-Lx/2)/Lx))*(1+cos(2*pi*(y-Ly/2)/Ly))/4;

      res = primitive_to_conservative((/dens, 0.0, vy, pres/))

   end function

   function quadrature(ax, bx, ay, by, xc, yc, t)result(r)

      real(PR), intent(in) :: ax, bx, ay, by, xc, yc, t

      real(PR), dimension(5) :: w, x

      real(PR), dimension(4) :: r

      real(PR) :: lx, cx, ly, cy

      integer :: i,j

      ! Gauss–Legendre quadrature (see https://en.wikipedia.org/wiki/Gaussian_quadrature)

      w(1) = (322.0-13.0*sqrt(70.0))/900.0;
      w(2) = (322.0+13.0*sqrt(70.0))/900.0;
      w(3) = 128.0/225.0;
      w(4) = (322.0+13.0*sqrt(70.0))/900.0;
      w(5) = (322.0-13.0*sqrt(70.0))/900.0;

      x(1) = - 1.0/3.0 * sqrt(5.0+2.0*sqrt(10.0/7.0));
      x(2) = - 1.0/3.0 * sqrt(5.0-2.0*sqrt(10.0/7.0));
      x(3) = 0.0;
      x(4) = + 1.0/3.0 * sqrt(5.0-2.0*sqrt(10.0/7.0));
      x(5) = + 1.0/3.0 * sqrt(5.0+2.0*sqrt(10.0/7.0));

      lx = 0.5*(bx - ax)
      cx = 0.5*(ax + bx)

      ly = 0.5*(by - ay)
      cy = 0.5*(ay + by)

      r =0.0

      do j = 1, 5
         do i = 1, 5
            r(:) = r(:) + w(j)*w(i)*f_isentropic_vortex(lx*x(i)+cx,ly*x(j)+cy, xc, yc, t)
         end do
      end do

      r = r*lx*ly


   end function quadrature


end module IC
