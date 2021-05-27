module parameters

  use constants
  ! Copy it as paramters.f90 in the main file

  implicit none

  ! Output files parameter
  character(100) :: file='output/'
  character(100) :: file_slice_x = 'output/Results/Sod/order3.dat'


  ! Time integration

  real(PR), parameter :: CFL  =  0.4
  integer , parameter :: time_method    = SSP_RK3
  logical , parameter :: dt_reduction = .false.

  ! Space integration
  integer,  parameter :: space_method   = GP_MOOD
  logical, parameter ::  cross_stencil = .false.
  logical, parameter ::  sphere_stencil = .true.
  integer , parameter :: Mord= 3  ! Order
  integer , parameter :: ngp = 1! Number of gaussian quadrature points per edges
  real(16), parameter :: l_16 = 3*(1./200)!/ell

  integer :: dim = 1


  ! Mesh parameter
  integer , parameter  :: ngc = 3 ! Number of ghost cells
  integer,  parameter :: lf = 200 ! Number of cell in the x direction
  integer,  parameter :: nf = 1  ! Number of cell in the y direction

  ! IC, BC and domain setup
  integer, parameter  :: IC_type = sodx
  real(PR), parameter :: tmax = 0.2
  real(16), parameter :: Lx_16 = 1. !Lenght of the domain in the x-direction
  real(16), parameter :: Ly_16 = Lx_16/lf!Lenght of the domain in the y-direction
  integer, parameter  :: BC_type = Neumann! Boundary conditions


  integer , parameter :: radius = (Mord -1)/2
  integer , parameter :: sz_sphere = (2*Mord - 1 + 4*(radius - 1)**2)
  integer , parameter :: sz_cross = 2*Mord-1

 ! MOOD Parameters, leave to true
  logical , parameter :: DMP        = .true.
  logical , parameter :: U2         = .true.
  logical , parameter :: U2_tol     = .true.

  ! Secondary variables , don't change
  real(16), parameter  :: dx_16 = Lx_16/lf, dy_16 = Ly_16/nf
  integer , parameter  :: lb = 1-ngc, le = lf + ngc, nb = 1-ngc, ne = nf + ngc
  real(PR), parameter :: dx = real(dx_16,PR), dy = real(dy_16,PR), Lx = real(Lx_16,PR), Ly = real(Ly_16,PR)

end module
