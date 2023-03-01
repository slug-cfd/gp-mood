module parameters

  use constants
  ! Copy it as paramters.f90 in the main file

  implicit none

  ! Output files parameter
  character(100) :: file='./plotter/Isentropic/isen_GP7_4quad_RK4_cfl0.8_HLLC_Lx20_200_ell_1.0_cross_'
  character(100) :: file_slice_x = './plotter/Isentropic/slice_x.dat'

  ! Time integration
  real(PR), parameter :: CFL  =  0.8
  integer , parameter :: time_method  = SSP_RK3
  logical , parameter :: dt_reduction = .false.

  integer, parameter :: space_method   = GP_MOOD
  logical, parameter :: cross_stencil  = .false.
  logical, parameter :: sphere_stencil = .true.
  integer, parameter :: Mord= 3  ! Order
  integer, parameter :: ngp = 2  ! Number of gaussian quadrature points per edges


  ! flux method
  integer, parameter :: numFlux = HLLC

  ! IO parameter
  integer, parameter :: IO_freqStep = -1    ! (put a positive number to use, e.g., 500)
  real(PR), parameter:: IO_freqTime = -1.0 ! (this is the default way to dump output files; put a positive number to use)


  integer :: dim = 2

  ! Mesh parameter
  integer , parameter :: ngc = 4 ! Number of ghost cells
  integer,  parameter :: lf = 256 ! Number of cell in the x direction
  integer,  parameter :: nf = 256 ! Number of cell in the y direction

  ! Set the baseline lf0 and nf0 for the dt reduction
  integer,  parameter :: lf0 = 50 ! Number of cell in the x direction
  integer,  parameter :: nf0 = 50 !
  

  ! IC, BC and domain setup
  integer, parameter  :: IC_type = isentropic_vortex
  real(PR), parameter :: tmax = 20. !20.
  integer, parameter  :: nmax = 1000000000 ! put a large number if want to finish based on tmax only
  real(16), parameter :: Lx_16 = 20.!20. !Lenght of the domain in the x-direction
  real(16), parameter :: Ly_16 = 20.!20. !Lenght of the domain in the y-direction
  integer, parameter  :: BC_type = Periodic ! Boundary conditions
  

  integer , parameter :: radius = (Mord -1)/2
  integer , parameter :: sz_sphere = min( (2*Mord - 1 + 4*(radius - 1)**2), 25)
  integer , parameter :: sz_cross = 2*Mord-1



 ! MOOD Parameters, leave to true; don't change 
  logical, save :: DMP
  logical , parameter :: U2         = .true.
  logical , parameter :: U2_tol     = .true.

  ! Secondary variables , don't change
  real(16), parameter :: dx_16 = Lx_16/lf, dy_16 = Ly_16/nf
  integer , parameter :: lb = 1-ngc, le = lf + ngc, nb = 1-ngc, ne = nf + ngc
  real(PR), parameter :: dx = real(dx_16,PR), dy = real(dy_16,PR), Lx = real(Lx_16,PR), Ly = real(Ly_16,PR)

  real(16), parameter :: l_16 = 6.*min(dx_16,dy_16) !/ell

!!$  if ((space_method == FOG) .and. (ngp .ne. 1)) then
!!$     print*, 'FOG should use ngp = 1'
!!$     stop
!!$  endif
  
end module parameters
