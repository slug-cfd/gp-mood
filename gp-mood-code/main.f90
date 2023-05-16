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
   use reader

   implicit none

   Real(PR) :: tic, tac, dt_sim, cfl_dt
   integer  :: IO_freqCounter = 1
   real(PR) :: nsteps

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

   if ( method ==  FOG ) then
      print*, 'Space method         =', 'First order godunov'
   else if(method == GP_SE) then
      print*, 'Space method         =', 'Linear GP - Squared exponential kernel'
      print*, 'Radius               =', (Mord-1)/2
      print*, '\ell                 =', real(l_16,4)
      print*, '\ell/dx              =', real(l_16/dx_16,4)
      print*, 'stencil shape      =', 'sphere'
   else if(method == GP_MOOD) then
      print*, 'Space method         = ', 'GP MOOD - Squared exponential kernel'
      print*, 'Radius =', (Mord-1)/2
      print*, '\ell =', real(l_16,8)
      print*, 'stencil shape      =', 'sphere'
      print*, '\ell/dx              =', real(l_16/dx_16,4)
   else if(method == Unlim_pol) then
      print*, 'Space method         = ', 'Unlimited polynomial reconstruction'
      print*, 'Radius               =', (Mord-1)/2
      print*, 'stencil shape      =', 'sphere'
   else if(method == POL_MOOD) then
      print*, 'Space method         =', 'polynomial MOOD'
      print*, 'Radius =', (Mord-1)/2
      print*, 'stencil shape      =', 'sphere'
   else if(method == NN_GP_MOOD) then
      print*, 'Space method         = ', 'NN GP MOOD - Squared exponential kernel'
      print*, 'Radius =', (Mord-1)/2
      print*, '\ell =', real(l_16,8)
      print*, 'stencil shape      =', 'sphere'
      print*, '\ell/dx              =', real(l_16/dx_16,4)
      print*, 'NN filename =', NN_filename

   else if(method == NN_GP_MOOD_CC) then
      print*, 'Space method         = ', 'NN GP MOOD Convex Combination - Squared exponential kernel'
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

   if ( integrator == FE ) then
      print*, 'time method          =', 'Forward_Euler'
   else if ( integrator == SSP_RK2 ) then
      print*, 'time method          = ', 'SSP_RK2'
   else if ( integrator == SSP_RK3 ) then
      print*, 'time method          = ', 'SSP_RK3'
   else if ( integrator == SSP_RK4 ) then
      print*, 'time method          = ', 'SSP_RK4'
   else
      print*, ' ERROR No time integrator detected'
      stop
   end if

   if ( dt_reduction ) then
      print*, ' The time step is reduced to match 5th order'
   end if
   
   call random_seed()

   call compute_metadata()
   print*, 'outputs name', file
   if ((method==NN_GP_MOOD).or.(method==NN_GP_MOOD_CC)) call load_NN()
  
   call init_mesh()
   call GP_presim_sphere()
   call pol_presim()

   print*,'----------------------------------------------------------------------------'

   call InitialCdt()
   call Boundary_C(U)

   call cpu_time (tic)

   niter = 0
   count_steps_NN_produced_NAN = 0
   count_correction = 0
   ! Restart
   if (restart) then
      call read()
      iter_0=niter+1
   else
      iter_0=1
   end if
   
   ! dump outout at t=0
   !call write_output(niter)

   dt_sim = dt

   do while ((t .lt. tmax) .and. (niter .le. nmax) )

      niter = niter + 1

      dtfinal = tmax - t

      call Setdt(U)

      cfl_dt = dt

      call time_stepping(U, Ur)

      call Boundary_C(U)

      ! update clock
      t =  t + dt

      U = Ur

      time(niter) = real(t,4)
      pct_detected_cell(niter) = real(count_detected_cell_RK*100/(nf*lf),4)

      if ((mod(niter,10) == 0) .or.(niter == 1)) then
         if (abs(cfl_dt - dt) > 0.) then
            print*,'nstep = ', niter, '|time = ',t,'|(dt, cfl_dt)=', dt,cfl_dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
         else
            print*,'nstep = ', niter, '|time = ',t,'|dt=', dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
         endif
         print*,' % of detected cell at the last iteration = ', pct_detected_cell(niter)
         print*,' % of a posteriori correction needed=', real(count_correction*100.0/(niter*nf*lf*3),4)
         if ((method==NN_GP_MOOD).or.(method==NN_GP_MOOD_CC))then 
            print*,' count_steps_NN_produced_NAN = ', count_steps_NN_produced_NAN
         end if
      end if


      ! dump output files based on the output frequency step
      if ((IO_freqStep > 0) .and. (mod(niter,IO_freqStep) == 0)) then
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
   print*,'time_spent_predicting = ', time_spent_predicting
   print*,'time_spent_correcting = ', time_spent_correcting
   print*,'time_spent_first_shot = ', time_spent_first_shot

   nsteps=(niter-iter_0+1)*nf*lf*1.0/1e6
   print*,'----------------MCell update per second = ', nsteps/res_time

   
   call write_output(niter)
   call write_diagnostic()
   if (write_NN_dataset) then 
      call write_NN_dataset_()
   end if



end program main
