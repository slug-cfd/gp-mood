module parameters

   use constants
   ! Copy it as paramters.f90 in the main file

   implicit none

   real(PR), parameter :: CFL  =  0.8
   integer , parameter :: integrator    = SSP_RK3
   logical , parameter :: restart = .true.
   character(100) :: restart_filename='output_2DRP4_GP_MOOD_CFL_0.8_512_512_100143.h5'
   
   integer, parameter :: method   = NN_GP_MOOD_CC
   real(16), parameter :: ell_o_dx = 12.0
   integer, parameter :: Mord= 3  ! Order
   integer, parameter :: ngp = 2  ! Number of gaussian quadrature points per edges

   logical , parameter :: DMP        = .true.
   logical , parameter :: U2         = .true.
   logical , parameter :: U2_tol     = .true.

   ! NN variables
   logical :: write_NN_dataset=.false.
   integer, parameter :: dataset_size = 1 ! Leave at one for running simu / high number for generating dataset
   integer, parameter :: L=57
   integer, parameter :: length=20
   character(100) :: NN_filename='model_2DRP4_512_first_20%_CEL_dropout_0.1_4_20'

   ! flux method
   integer, parameter :: numFlux = HLLC

   ! IO parameter
   integer, parameter :: IO_freqStep = -100    ! (put a positive number to use, e.g., 500)
   real(PR), parameter:: IO_freqTime = -1.e-3 ! (this is the default way to dump output files; put a positive number to use)

   ! Mesh parameter
   integer,  parameter :: lf = 512 ! Number of cell in the x direction
   integer,  parameter :: nf = 512  ! Number of cell in the y direction

   ! IC, BC and domain setup
   integer, parameter  :: problem = RP_2D_4
   real(PR), parameter :: tmax = 0.25
   integer, parameter  :: nmax = 99999 ! put a large number if want to finish based on tmax only
   real(16), parameter :: Lx_16 = 1. !Lenght of the domain in the x-direction
   real(16), parameter :: Ly_16 = 1. !Lenght of the domain in the y-direction
   integer, parameter  :: BC_type = Neumann ! Boundary conditions
   
end module
