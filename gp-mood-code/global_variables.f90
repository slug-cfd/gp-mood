module global_variables
   use parameters

   implicit none


   ! Set the baseline lf0 and nf0 for the dt reduction
   integer,  parameter :: lf0 = 400 ! Number of cell in the x direction
   integer,  parameter :: nf0 = 400 !
   integer , parameter :: ngc = 4 ! Number of ghost cells

   integer , dimension(7), parameter :: szs_sphere = (/1,0,5,0,13,0,25/)
 
   integer , parameter :: sz_sphere = szs_sphere(Mord)
   integer , parameter :: sz_sphere_p1 = szs_sphere(Mord+2)

   integer , parameter :: radius = (Mord -1)/2

   ! Secondary variables , don't change
   real(16), parameter :: dx_16 = Lx_16/lf, dy_16 = Ly_16/nf
   integer , parameter :: lb = 1-ngc, le = lf + ngc, nb = 1-ngc, ne = nf + ngc
   real(PR), parameter :: dx = real(dx_16,PR), dy = real(dy_16,PR), Lx = real(Lx_16,PR), Ly = real(Ly_16,PR)
   real(16), parameter :: l_16 = ell_o_dx*min(dx_16,dy_16) !/ell

   ! Solution
   real(PR), dimension(4,lb:le, nb:ne ), save :: U = 0., Ur = 0.

   ! High Riemann order states
   real(PR), dimension(4, iL:iB, ngp, lb:le, nb:ne), save :: Uh = 0.

   ! Time and time steps
   real(PR), save :: t = 0., dt = 1e30, dtfinal = 1e30
   real(PR), save     :: res_time, t_rk= 0.

   Real(PR) :: error_inversion

   ! Mesh
   real(PR), dimension(lb:le), save         :: mesh_x = 0.
   real(PR), dimension(nb:ne), save         :: mesh_y = 0.

   !spherical GP coefficients
   real(PR),dimension(Mord, sz_sphere,iL:iB,ngp), save :: zT_sp
   real(PR),dimension(Mord, sz_sphere), save           :: GP_d2x_sp, GP_d2y_sp

   ! Pol Mood coefficients
   real(PR),dimension(2*3-1,iL:iB,2), save :: Pol_zT_o3
   real(PR),dimension(2*5-1,iL:iB,2), save :: Pol_zT_o5

   !pol coefficients
   real(PR),dimension(Mord, sz_sphere,iL:iB,ngp), save :: Pol_zT

   !GP indices
   integer, dimension(7, 25,2), save  :: ixiy_sp
   integer, dimension(7+2, 25,2), save  :: ixiy_sp1

   !MOOD variables
   logical, save                                :: MOOD_finished

   integer, dimension(lb:le, nb:ne), save       :: CellGPO, CellGPO_MOOD
   logical, dimension(lb:le, nb:ne), save       :: DetCell

   integer, dimension(-1:lf+1,  1:nf  ), save   :: FaceGPO_x
   integer, dimension( 1:lf  , -1:nf+1), save   :: FaceGPO_y

   logical, dimension(-1:lf+1,  0:nf+1  ), save :: DetFace_x
   logical, dimension( 0:lf+1  , -1:nf+1), save :: DetFace_y

   real(PR)   , dimension(5,5), save :: gauss_weight

   integer, save  :: niter, count_steps_NN_produced_NAN
   Real(PR), save :: count_detected_cell_RK, count_NN_PAD_RK

   ! NN variables
   real(4), dimension(lenght, L     ) :: weight0=-6666666
   real(4), dimension(lenght, lenght) :: weight1=-6666666
   real(4), dimension(2     , lenght) :: weight2=-6666666

   real(4), dimension(lenght, 1) :: bias0=-6666666
   real(4), dimension(lenght, 1) :: bias1=-6666666
   real(4), dimension(2     , 1) :: bias2=-6666666

   integer, parameter :: size_weight0 = lenght*L
   integer, parameter :: size_weight1 = lenght*lenght
   integer, parameter :: size_weight2 = 2*lenght

   integer, parameter :: size_bias0 = lenght
   integer, parameter :: size_bias1 = lenght
   integer, parameter :: size_bias2 = 2

   integer, parameter :: up0 = size_weight0
   integer, parameter :: up1 = up0 + size_bias0
   integer, parameter :: up2 = up1 + size_weight1
   integer, parameter :: up3 = up2 + size_bias1
   integer, parameter :: up4 = up3 + size_weight2
   integer, parameter :: up5 = up4 + size_bias2

   !Diagnostic
   Real(4), dimension(nmax) :: time = -666
   Real(4), dimension(nmax) :: pct_detected_cell = -666

   ! Metadata 
   character(3)  :: CFL_char
   character(10) :: problem_char
   character(10) :: method_char
   character(100) :: file

   !Training dataset
   integer, parameter :: dataset_size = 100000
   real(4), dimension(dataset_size, L) :: inputs=-666
   real(4), dimension(dataset_size, 2) :: labels=-666
   real(4) :: NR0=0, NR1=0, freq_R0=0.5, freq_R0_target=0.5
   integer :: index=1, n_overwrite=0

end module
