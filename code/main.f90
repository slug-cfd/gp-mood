program main
  use parameters
  use mesh_init
  use IC
  use BC
  use time_stepper
  use global_variables
  use output
  use GP_init


  implicit none

  Real(PR) :: tic, tac, dt_sim, cfl_dt
  integer  :: IO_freqCounter = 1
  

  if (cross_stencil .and. sphere_stencil) then
     print*, 'both shape selected'
     stop
  end if


  print*,'----------------------------------------------------------------------------'
  print*,'Simulation parameters : '
  print*, 'lf                      =', lf
  print*, 'nf                      =', nf
  print*, 'CFL                     =', Real(CFL,4)
  print*, 'Lx                      =', Real(Lx,4)
  print*, 'Ly                      =' ,Real(Ly,4)
  print*, 'Number of ghost cells   =', ngc

  if (space_method  == GP_MOOD) then
     if (sphere_stencil) then
        print*, 'Number of sphere stencil=', sz_sphere
     else
        print*, 'Number of cross stencil = ', sz_cross
     end if
  endif

  if ( IC_type == Sodx ) then
     print*, 'Problem                =', 'Sod shock tube'
  else if ( IC_type == Sod_rotated ) then
     print*, 'Problem                =', 'Rotated Sod shock tube'
  else if ( IC_type == Lax) then
     print*, 'Problem              =', 'Lax shock tube'
  else if ( IC_type == Shu_Osher) then
     print*, 'Problem              =', 'Shu Osher shock tube'
  else if ( IC_type == Shu_Osher_rotated) then
     print*, 'Problem              =', 'Shu Osher shock tube'
  else if ( IC_type == strong_raref) then
     print*, 'Problem              =','Strong rarefaction'
  else if ( IC_type == BLAST) then
     print*, 'Problem              =','BLAST interaction'
  else if ( IC_type == Lin_Gauss_xy) then
     print*, 'Problem              =','Advection of a gaussian along the xy axis'
  else if ( IC_type == KH) then
     print*, 'Problem              =','Kevin Helmoltz instability'
  else if ( IC_type == RP_2D_3) then
     print*, 'Problem              =', '2D Riemann problem, config 3'
  else if ( IC_type == RP_2D_12) then
     print*, 'Problem              =', '2D Riemann problem, config 12'
  else if ( IC_type == DMR) then
     print*, 'Problem              =', 'double mach reflection'
  else if ( IC_type == implosion) then
     print*, 'Problem              =', 'implosion'
  else if ( IC_type == isentropic_vortex) then
     print*, 'Problem              =', 'isentropic_vortex'
  else if ( IC_type == sedov) then
     print*, 'Problem              =', 'sedov test'
  else if ( IC_type == Lin_Gauss_x) then
     print*, 'Problem              =', 'Advection of a gaussian along the x axis'
  else if ( IC_type == Mach800) then
     print*, 'Problem              =', 'Mach800 jet in y-direction'
  else if ( IC_type == DoubleMach800) then
     print*, 'Problem              =', 'Double Mach800 jets collision in y-direction'
  else
     print*, 'problem not discussed in the paper'
     stop
  end if

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
     if ( cross_stencil ) then
        print*, 'stencil shape      =', 'cross'
     else if ( sphere_stencil) then
        print*, 'stencil shape      =', 'sphere'
     else
        print*, ' no stencil picked'
        stop
     end if
  else if(space_method == GP_MOOD) then
     print*, 'Space method         = ', 'GP MOOD - Squared exponential kernel'
     print*, 'Radius =', (Mord-1)/2
     print*, '\ell =', real(l_16,8)
     if ( cross_stencil ) then
        print*, 'stencil shape      =', 'cross'
     else if ( sphere_stencil) then
        print*, 'stencil shape      =', 'sphere'
     else
        print*, ' no stencil picked'
        stop
     end if
     print*, '\ell/dx              =', real(l_16/dx_16,4)
  else if(space_method == Unlim_pol) then
     print*, 'Space method         = ', 'Unlimited polynomial reconstruction'
     print*, 'Radius               =', (Mord-1)/2
     if ( cross_stencil ) then
        print*, 'stencil shape      =', 'cross'
     else if ( sphere_stencil) then
        print*, 'stencil shape      =', 'sphere'
     else
        print*, ' no stencil picked'
        stop
     end if
  else if(space_method == POL_MOOD) then
     print*, 'Space method         =', 'polynomial MOOD'
     print*, 'Radius =', (Mord-1)/2
     if ( cross_stencil ) then
        print*, 'stencil shape      =', 'cross'
     else if ( sphere_stencil) then
        print*, 'stencil shape      =', 'sphere'
     else
        print*, ' no stencil picked'
        stop
     end if
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


  call init_mesh()
  if (cross_stencil) call GP_presim()
  if (sphere_stencil)call GP_presim_sphere()

  call pol_presim()

  print*,'----------------------------------------------------------------------------'

  call InitialCdt()
  call Boundary_C(U)

  call cpu_time (tic)

  niter = 0

  ! dump outout at t=0
  call write_output(niter)

  dt_sim = min(1.e-10,dt)

  do while ((t .lt. tmax) .and. (niter .le. nmax) )

     niter = niter + 1

     dtfinal = tmax - t


     call Setdt(U, niter)
     !print*,'CFL dt =', dt
     cfl_dt = dt
     call time_stepping(U, Ur)
     call Boundary_C(U)

     
     if (dt_sim < dt) then
        t = t + dt_sim
        dt_sim = 2.*dt_sim
        !force small dt
        dt = dt_sim
     else
        t =  t + dt
     endif

     U = Ur
     if ((mod(niter,10) == 0) .or.(niter == 1)) then
        if (abs(cfl_dt - dt) > 0.) then
           print*,'nstep = ', niter, '|time = ',t,'|(dt, cfl_dt)=', dt,cfl_dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
        else
           print*,'nstep = ', niter, '|time = ',t,'|dt=', dt, '|' , real(100*(tmax-t)/tmax,4),'% done'
        endif
        print*,' % of detected cell at the last iteration = ', real(count_FE*100/(nf*lf),4)
        !! DL -- dump outputs regularly, say, every 100 step
     end if


     !print*,niter, mod(niter,100)
     ! dump output files based on the output frequency step
     if ((IO_freqStep > 0) .and. (mod(niter,IO_freqStep) == 0)) then
        print*,''
        print*,'======================================================================'
        print*,'   a new output has been written, file number=',niter
        print*,'======================================================================'
        print*,''
        call write_output(niter)
     endif

     ! dump output files based on the output frequency time interval
     if (IO_freqTime > 0.) then
        if ((t     -real(IO_freqCounter)*IO_freqTime < 0.) .and. &
            (t+dt - real(IO_freqCounter)*IO_freqTime > 0.)) then

           IO_freqCounter = IO_freqCounter + 1
           print*,''
           print*,'======================================================================'
           print*,'   a new output has been written, file number=',niter
           print*,'======================================================================'
           print*,''
           call write_output(niter)           
        endif
     endif

  end do

  call cpu_time (tac)

  res_time = tac - tic
  print*,'time = ',t,'dt=', dt
  print*,'Res time = ', tac - tic

  call write_slice(dir_x)
  call write_slice(dir_y)
  call write_slice(dir_xy)
  call write_slice(dir_yx)


  print*,''
  print*,'======================================================================'
  print*,'   a new output has been written, file number=',niter
  print*,'======================================================================'
  print*,''
  call write_output(niter)
  
  call error()


end program main
