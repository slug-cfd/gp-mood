module constants
  implicit none


  integer  , parameter :: PR = 8 ! Precision

  !Maths constants
  real(16) , parameter :: pi_16 = 4*atan(1._16)
  real(PR) , parameter :: pi    = 4*atan(1.   )    , y     = 1.4,  eps     = 1.0e-31, sixth = 1./6, quarter = 0.25, thirtwe = 13./12
  real(16) , parameter :: ospi  = 1._16/sqrt(pi_16)
  real(16) , parameter :: sq3   = sqrt(3._16),  sq35 = sqrt(3._16/5), sq2=sqrt(2._16)

  real(PR) , parameter :: sq65 = sqrt(6./5), sq30 = sqrt(30.), third = 1./3, sq107 = sqrt(10./7), sq70 = sqrt(70.)


  !Time Methods indices
  integer, parameter :: FOG     = 1, GP_MOOD = 2, POL_MOOD = 200, Unlim_POL = 201, GP_SE = 202
  !Space Methods indices
  integer, parameter :: FE      = 8, SSP_RK2 = 9, SSP_RK3 =10, SSP_RK4 =12
  !Flux methods
  integer, parameter :: HLLC = 100, LLF = 101, HLL = 102

  !BC indices
  integer, parameter :: Neumann =17, Periodic =18, Dirichlet =19, Reflective =20
  integer, parameter :: Mach800_BC = 201, DoubleMach800_BC = 301, RMI_BC = 401
  !IC indices
  integer, parameter :: SODx        =21, SODy             =22, SODxy        =23, explosion =36
  integer, parameter :: Lin_Gauss_x =24, Lin_Gauss_y      =25, Lin_Gauss_xy =26, Shu_Osher =27
  integer, parameter :: RP_2D_3     =28, implosion        =29, RP_2D_12     =30, DMR       =35
  integer, parameter :: RP_2D_15    =31, strong_raref     =32, Lax          =33, BLAST     =34
  integer, parameter :: RP_2D_6     =38, sedov            =39, M3WT         =40, KH        =41
  integer, parameter :: isentropic_vortex = 42, sod_rotated = 43, Shu_Osher_rotated = 44
  integer, parameter :: Mach800 = 45, DoubleMach800 = 46, RMI = 47, SlowShock = 48

  !Data indices
  integer, parameter :: rho   = 1, momx  = 2, momy   = 3, ener = 4
  integer, parameter :: dir_x = 1, dir_y = 2, dir_xy = 3, dir_yx = 4
  integer, parameter :: iL    = 1, iT    = 2, iR     = 3, iB   = 4

  !finite differences
  real, parameter, dimension(3)  :: D2O2 = (/1., -2., 1./)
  real, parameter, dimension(5)  :: D2O4 = (/-1./12, 4./3, -5./2, 4./3, -1./12  /)
  real, parameter, dimension(7)  :: D2O6 = (/1./90, -3./20, 3./2, -49./18, 3./2, -3./20, 1./90  /)
  real, parameter, dimension(9)  :: D2O8 = (/-1./560, 8./315, -1./5, 8./5, -205./72, 8./5, -1./5, 8./315, -1./560/)


  !Constants for SSP_RK4
  real(PR) :: a10 = 1.               ,                          c1 = 0.391752226571890
  real(PR) :: a20 = 0.444370493651235, a21 = 0.555629506348765, c2 = 0.368410593050371
  real(PR) :: a30 = 0.620101851488403, a32 = 0.379898148511597, c3 = 0.251891774271694
  real(PR) :: a40 = 0.178079954393132, a43 = 0.821920045606868, c4 = 0.544974750228521
  real(PR) :: f2  = 0.517231671970585, f3  = 0.096059710526147, f4 = 0.386708617503269
  real(PR) :: ff3 = 0.063692468666290, ff4 =0.226007483236906

  !Constants for Pol MOOD

  integer :: kM=1, kCloseNrm=2, kCloseTrans=3, kFarTrans = 4, kFarNrm = 5, kCloseNrm2 = 6, kCloseTrans2=7, kFarTrans2=8, kFarNrm2=9

  real(PR), dimension(5) :: o3_coef   = (/15./18, 1./3, 1./(4*sqrt(3.)), -1./(4*sqrt(3.)), -1./6 /)


end module
