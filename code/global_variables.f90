module global_variables
  use parameters
  implicit none

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


  !cross GP coefficients
  real(PR),dimension(Mord, sz_cross,iL:iB,ngp), save :: zT
  real(PR),dimension(Mord, sz_cross), save           :: GP_d2x, GP_d2y

  !spherical GP coefficients
  real(PR),dimension(Mord, sz_sphere,iL:iB,ngp), save :: zT_sp
  real(PR),dimension(Mord, sz_sphere), save           :: GP_d2x_sp, GP_d2y_sp


  ! Pol Mood coefficients
  real(PR),dimension(2*3-1,iL:iB,2), save :: Pol_zT_o3
  real(PR),dimension(2*5-1,iL:iB,2), save :: Pol_zT_o5


  !pol coefficients
  real(PR),dimension(Mord, sz_cross,iL:iB,ngp), save :: Pol_zT

  !GP indices
  integer, dimension(Mord, sz_cross,2), save   :: ixiy
  integer, dimension(Mord, sz_sphere,2), save  :: ixiy_sp



  !MOOD variables
  logical, save                                :: MOOD_finished

  integer, dimension(lb:le, nb:ne), save       :: CellGPO
  logical, dimension(lb:le, nb:ne), save       :: DetCell

  integer, dimension(-1:lf+1,  1:nf  ), save   :: FaceGPO_x
  integer, dimension( 1:lf  , -1:nf+1), save   :: FaceGPO_y

  logical, dimension(-1:lf+1,  0:nf+1  ), save :: DetFace_x
  logical, dimension( 0:lf+1  , -1:nf+1), save :: DetFace_y



  real(PR)   , dimension(ngp,ngp), save :: gauss_weight

  integer, save  :: niter,count_detectio
  Real(PR), save :: count_FE, count_RK


end module
