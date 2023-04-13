module physics
   use constants
   use parameters
   use global_variables
   implicit none

contains

   function Flux(u, dir, vx, vy, vx2, vy2, pres, adm)result(r)
      !=========== Compute the flux function of the system ===========!
      ! Always in conservative variables

      !============= Input(s) ==============!
      real(PR)   , dimension(4), intent(in)  :: u
      integer,               intent(in)  :: dir
      real(PR)   ,           intent(in)  :: vx, vy, vx2, vy2, pres

      logical, optional, intent(in)      :: adm

      !============= Output(s) =============!
      real(PR), dimension(4)              :: r
      !============= Instructions =============!


      if ((present(adm).and.adm).or.(.not.present(adm))) then

         if (dir == dir_x) then
            r(1) = u(momx)
            r(2) = u(rho )*vx2 + pres

            r(3) = u(momy)*vx

            r(4) = vx*(u(ener) + pres )
         else if (dir == dir_y) then
            r(1) = u(momy)
            r(2) = u(momx)*vy
            r(3) = u(rho )*vy2 + pres
            r(4) = vy*(u(ener) + pres )
         else
            print*, 'dir =/ x or y'
            stop
         end if

      else
         r(:) = 1./(vx-vx) !DL -- enforcing NaN here to be detected.
      end if

   end function Flux


   subroutine c2p_local(u, vx, vy, vx2, vy2, pres, c, adm)
      !============= Input(s) ==============!
      real(PR), dimension(4), intent(in)  :: u

      real(PR), intent(out)          :: vx, vy, vx2, vy2, pres, c
      logical, optional, intent(inout) :: adm



      !============ Instructions ===========!
      vx   =  u(momx)/u(rho)
      vx2  =  vx**2

      vy   =  u(momy)/u(rho)
      vy2  =  vy**2


      pres =  pressure(u)
      c    =  sqrt(y*pres/u(rho))

      if (present(adm)) then
         if ((u(rho)<=0.).or.(pres <=0).or.(ISNAN(u(rho))).or.(ISNAN(pres))) adm = .false.
      end if

   end subroutine c2p_local


   subroutine Setdt(Uin)
      !============= Variables ==============!
      integer            :: l, n

      real(PR) ::  vx, vy, vx2, vy2, pres, c
      !real(PR) ::  vx, vy, vx2, vy2, pres, c, dx0 =  0.31250000000000000*(Lx/10), dy0 =0.31250000000000000*(Lx/10)
      real(PR) :: dx0, dy0

      real(PR), dimension(4,lb:le, nb:ne ), intent(in) :: Uin

      !============ Instructions ===========!

      dt = 1.0e30

      dx0 = Lx/lf0
      dy0 = Ly/nf0

      !!DL -- Include the guardcell information for dt calculation
!!$    do n = 1, nf
!!$       do l = 1, lf
      do n = 1-ngc, nf+ngc
         do l = 1-ngc, lf+ngc

            call c2p_local(Uin(:,l,n), vx, vy, vx2, vy2, pres, c)


            !if (dt_reduction)  then
            !   dt = min(dt,  min( dx0/(abs(vx)+c), dy0/(abs(vy)+c) ))
            !else
            !   dt = min(dt,  min( dx/(abs(vx)+c), dy/(abs(vy)+c) ))
            !end if

            dt = min(dt,  1.0/( 1.0/(dx/(abs(vx)+c)) + 1.0/(dy/(abs(vy)+c)) ) ) 

         end do
      end do


      if ((dt_reduction)) then
         dt = dt*(dx/dx0)**(real(Mord)/4 )
      end if

      dt = dt *CFL

      if (dt > dtfinal) then
         dt = dtfinal
      end if




   end subroutine Setdt


   function pressure(u)result(p)
      !============= Input(s) ==============!
      real(PR), dimension(4), intent(in)  :: u

      !============= Output(s) =============!
      real(PR)                            :: p

      !============ Instructions ===========!

      p   = (y-1.)*(u(ener)-(0.5*( U(momx)**2 + U(momy)**2 )/U(rho)) )

   end function pressure


   function primitive_to_conservative(v)result(u)
      !============= Input(s) ==============!
      real(PR), dimension(4), intent(in)  :: v

      !============= Output(s) =============!
      real(PR), dimension(4)              :: u

      !============ Instructions ===========!

      u(rho ) =  v(1)
      u(momx) =  v(1)*v(2)
      u(momy) =  v(1)*v(3)

      u(ener)= ( (v(4)/(y-1.)) + 0.5*v(1)*(v(2)**2+v(3)**2) )
   end function primitive_to_conservative

   function conservative_to_primitive(u)result(v)
      !============= Input(s) ==============!
      real(PR), dimension(4), intent(in)  :: u

      !============= Output(s) =============!
      real(PR), dimension(4)              :: v

      !============ Instructions ===========!

      v(1) =  u(rho)
      v(2) =  u(momx)/u(rho)
      v(3) =  u(momy)/u(rho)
      v(4) =  pressure(u)
   end function conservative_to_primitive
end module physics
