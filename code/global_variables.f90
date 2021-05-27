module global_variables
  use parameters
  implicit none

  ! Solution
  real(PR), dimension(4,lb:le, nb:ne )             :: U = 0., Ur = 0.

  ! High Riemann order states
  real(PR), dimension(4, iL:iB, ngp, lb:le, nb:ne) :: Uh = 0.


  ! Time and time steps
  real(PR) :: t = 0., dt = 1e30, dtfinal = 1e30
  real(PR)     :: res_time, t_rk= 0.

  Real(PR) :: error_inversion

  ! Mesh
  real(PR), dimension(lb:le)         :: mesh_x = 0.
  real(PR), dimension(nb:ne)         :: mesh_y = 0.


  !cross GP coefficients
  real(PR),dimension(Mord, sz_cross,iL:iB,ngp) :: zT
  real(PR),dimension(Mord, sz_cross)            :: GP_d2x, GP_d2y

  !spherical GP coefficients
  real(PR),dimension(Mord, sz_sphere,iL:iB,ngp) :: zT_sp
  real(PR),dimension(Mord, sz_sphere)            :: GP_d2x_sp, GP_d2y_sp


  ! Pol Mood coefficients
  real(PR),dimension(2*3-1,iL:iB,2) :: Pol_zT_o3
  real(PR),dimension(2*5-1,iL:iB,2) :: Pol_zT_o5


  !pol coefficients
  real(PR),dimension(Mord, sz_cross,iL:iB,ngp) :: Pol_zT

  !GP indices
  integer, dimension(Mord, sz_cross,2)          :: ixiy
  integer, dimension(Mord, sz_sphere,2)         :: ixiy_sp



  !MOOD variables
  logical                                :: MOOD_finished

  integer, dimension(lb:le, nb:ne)       :: CellGPO
  logical, dimension(lb:le, nb:ne)       :: DetCell

  integer, dimension(-1:lf+1,  1:nf  )   :: FaceGPO_x
  integer, dimension( 1:lf  , -1:nf+1)   :: FaceGPO_y

  logical, dimension(-1:lf+1,  0:nf+1  ) :: DetFace_x
  logical, dimension( 0:lf+1  , -1:nf+1) :: DetFace_y



  real(PR)   , dimension(ngp,ngp) :: gauss_weight

  integer :: niter,count_detectio
  Real(PR) :: count_FE, count_RK


end module
