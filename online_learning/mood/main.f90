program main
   use parameters
   use mesh_init
   use IC
   use BC
   use time_stepper
   use global_variables
   use output
   use GP_init
   use mod_NN


   implicit none

   Real(PR) :: tic, tac, dt_sim, cfl_dt
   integer  :: IO_freqCounter = 1

   print*,'----------------------------------------------------------------------------'
   print*,'Simulation parameters : '
   print*, 'lf                      =', lf
   print*, 'nf                      =', nf
   print*, 'CFL                     =', Real(CFL,4)
   print*, 'Lx                      =', Real(Lx,4)
   print*, 'Ly                      =' ,Real(Ly,4)
   print*, 'Number of ghost cells   =', ngc

   print*, 'size of sphere stencil=', sz_sphere


   if ( BC_type == Neumann ) then
      print*, 'Boundary Condition   =','Neumann'
   else if( BC_type == Periodic ) then
      print*, 'Boundary Condition   =','Periodic'
   else if( BC_type == Reflective ) then
      print*, 'Boundary Condition   =','Reflective'
   else if( BC_type == DMR) then
      print*, 'Boundary Condition   =','Double Mach reflection'
   end if

   if ( space_method ==  FOG ) then
      print*, 'Space method         =', 'First order godunov'
   else if(space_method == GP_SE) then
      print*, 'Space method         =', 'Linear GP - Squared exponential kernel'
      print*, 'Radius               =', (Mord-1)/2
      print*, '\ell                 =', real(l_16,4)
      print*, '\ell/dx              =', real(l_16/dx_16,4)
      print*, 'stencil shape      =', 'sphere'
   else if(space_method == GP_MOOD) then
      print*, 'Space method         = ', 'GP MOOD - Squared exponential kernel'
      print*, 'Radius =', (Mord-1)/2
      print*, '\ell =', real(l_16,8)
      print*, 'stencil shape      =', 'sphere'
      print*, '\ell/dx              =', real(l_16/dx_16,4)
   else if(space_method == Unlim_pol) then
      print*, 'Space method         = ', 'Unlimited polynomial reconstruction'
      print*, 'Radius               =', (Mord-1)/2
      print*, 'stencil shape      =', 'sphere'
   else if(space_method == POL_MOOD) then
      print*, 'Space method         =', 'polynomial MOOD'
      print*, 'Radius =', (Mord-1)/2
      print*, 'stencil shape      =', 'sphere'
   else if(space_method == NN_GP_MOOD) then
      print*, 'Space method         = ', 'NN GP MOOD - Squared exponential kernel'
      print*, 'Radius =', (Mord-1)/2
      print*, '\ell =', real(l_16,8)
      print*, 'stencil shape      =', 'sphere'
      print*, '\ell/dx              =', real(l_16/dx_16,4)
      print*, 'NN filename =', NN_filename

   else if(space_method == eval_NN_GP_MOOD) then
      print*, 'Space method         = ', 'eval_NN GP MOOD - Squared exponential kernel'
      print*, 'Radius =', (Mord-1)/2
      print*, '\ell =', real(l_16,8)
      print*, 'stencil shape      =', 'sphere'
      print*, '\ell/dx              =', real(l_16/dx_16,4)
      print*, 'NN filename =', NN_filename
   else
      print*, 'space method not programmed'
      stop
   end if

   print*, ' Number au gaussian point per edge =', ngp

   if ( time_method == FE ) then
      print*, 'time method          =', 'Forward_Euler'
   else if ( time_method == SSP_RK2 ) then
      print*, 'time method          = ', 'SSP_RK2'
   else if ( time_method == SSP_RK3 ) then
      print*, 'time method          = ', 'SSP_RK3'
   else if ( time_method == SSP_RK4 ) then
      print*, 'time method          = ', 'SSP_RK4'
   else
      print*, ' ERROR No time integrator detected'
      stop
   end if

   if ( dt_reduction ) then
      print*, ' The time step is reduced to match 5th order'
   end if

   print*, 'output directory:', file

   if ((space_method==NN_GP_MOOD).or.(space_method==eval_NN_GP_MOOD)) call load_NN(NN_filename)
  
   call init_mesh()
   call GP_presim_sphere()
   call pol_presim()

   print*,'----------------------------------------------------------------------------'

   call InitialCdt()
   call Boundary_C(U)

   call cpu_time (tic)

   niter = 0

   ! dump outout at t=0
   call write_output(niter)

   dt_sim = dt

   do while ((t .lt. tmax) .and. (niter .le. nmax) )


      niter = niter + 1

      dtfinal = tmax - t


      call Setdt(U)
      !print*,'CFL dt =', dt
      cfl_dt = dt

      call time_stepping(U, Ur)
      call Boundary_C(U)

      ! update clock
      t =  t + dt

      U = Ur
      if ((mod(niter,1) == 0) .or.(niter == 1)) then
         if (abs(cfl_dt - dt) > 0.) then
            print*,'nstep = ', niter, '|time = ',t,'|(dt, cfl_dt)=', dt,cfl_dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
         else
            print*,'nstep = ', niter, '|time = ',t,'|dt=', dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
         endif
         print*,' % of detected cell at the last iteration = ', real(count_detected_cell_RK*100/(nf*lf),4)
         !! DL -- dump outputs regularly, say, every 100 step
      end if


      !print*,niter, mod(niter,100)
      ! dump output files based on the output frequency step
      if ((IO_freqStep > 0) .and. (mod(niter,IO_freqStep) == 0)) then
         print*,''
         print*,'======================================================================'
         print*,'   A new output has been written, file number=',niter
         print*,'   Output directory:', file
         print*,'======================================================================'
         print*,''
         call write_output(niter)
      endif

      ! dump output files based on the output frequency time interval
      if (IO_freqTime > 0.) then
         if ((t     -real(IO_freqCounter)*IO_freqTime < 0.) .and. &
            (t+dt - real(IO_freqCounter)*IO_freqTime > 0.)) then

            IO_freqCounter = IO_freqCounter + 1
            call write_output(niter)
         endif
      endif

   end do

   call cpu_time (tac)

   res_time = tac - tic
   print*,'time = ',t,'dt=', dt
   print*,'Res time = ', tac - tic

   call write_output(niter)



end program main
