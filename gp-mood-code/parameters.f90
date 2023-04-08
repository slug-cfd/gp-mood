module parameters

   use constants
   ! Copy it as paramters.f90 in the main file

   implicit none

   ! Output files parameter
   character(100) :: file='output_sedov_POLMOOD_FE'

   logical :: write_NN_dataset=.false.

   real(PR), parameter :: CFL  =  0.8
   integer , parameter :: time_method    = FE
   logical , parameter :: dt_reduction = .false.

   integer, parameter :: space_method   = GP_MOOD
   integer, parameter :: Mord= 3  ! Order
   integer, parameter :: ngp = 2  ! Number of gaussian quadrature points per edges

   logical , parameter :: DMP        = .true.
   logical , parameter :: U2         = .true.
   logical , parameter :: U2_tol     = .true.

   ! flux method
   integer, parameter :: numFlux = HLLC

   ! IO parameter
   integer, parameter :: IO_freqStep = 1    ! (put a positive number to use, e.g., 500)
   real(PR), parameter:: IO_freqTime = -1.e-3 ! (this is the default way to dump output files; put a positive number to use)

   ! Mesh parameter
   integer , parameter :: ngc = 4 ! Number of ghost cells
   integer,  parameter :: lf = 128 ! Number of cell in the x direction
   integer,  parameter :: nf = 128  ! Number of cell in the y direction

   ! Set the baseline lf0 and nf0 for the dt reduction
   integer,  parameter :: lf0 = 400 ! Number of cell in the x direction
   integer,  parameter :: nf0 = 400 !


   ! IC, BC and domain setup
   integer, parameter  :: IC_type = sedov
   real(PR), parameter :: tmax = 0.05
   integer, parameter  :: nmax = 999999 ! put a large number if want to finish based on tmax only
   real(16), parameter :: Lx_16 = 1. !Lenght of the domain in the x-direction
   real(16), parameter :: Ly_16 = 1. !Lenght of the domain in the y-direction
   integer, parameter  :: BC_type = Neumann ! Boundary conditions
   
end module
