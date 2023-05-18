module parameters

   use constants
   ! Copy it as paramters.f90 in the main file

   implicit none

   real(PR), parameter :: CFL  =  0.8
   integer , parameter :: integrator    = SSP_RK3
   logical , parameter :: restart = .false.
   character(100) :: restart_filename='.'

   integer, parameter :: method   = GP_MOOD
   real(16), parameter :: ell_o_dx = 12.0
   integer, parameter :: Mord= 3  ! Order
   integer, parameter :: ngp = 2  ! Number of gaussian quadrature points per edges

   logical , parameter :: DMP        = .false.
   logical , parameter :: U2         = .false.
   logical , parameter :: U2_tol     = .false.

   ! NN variables
   logical :: write_NN_dataset=.false.
   integer, parameter :: dataset_size = 1 ! Leave at one for running simu / high number for generating dataset
   integer, parameter :: L=57
   integer, parameter :: length=60
   character(100) :: NN_filename=''

   ! flux method
   integer, parameter :: numFlux = unsplit

   ! IO parameter
   integer, parameter :: IO_freqStep = -100   ! (put a positive number to use, e.g., 500)
   real(PR), parameter:: IO_freqTime = -1.e-3 ! (this is the default way to dump output files; put a positive number to use)

   ! Mesh parameter
   integer,  parameter :: lf = 256 ! Number of cell in the x direction
   integer,  parameter :: nf = 256  ! Number of cell in the y direction

   ! IC, BC and domain setup
   integer, parameter  :: problem = RP_2D_3
   real(PR), parameter :: tmax = 0.8
   integer, parameter  :: nmax = 9999 ! put a large number if want to finish based on tmax only
   real(16), parameter :: Lx_16 = 1. !Lenght of the domain in the x-direction
   real(16), parameter :: Ly_16 = 1. !Lenght of the domain in the y-direction
   integer, parameter  :: BC_type = Neumann ! Boundary conditions
   
end module