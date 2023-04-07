module secondary_parameters

    use constants
    use parameters

    integer , dimension(7), parameter :: szs_sphere = (/1,0,5,0,13,0,25/)
 
    integer , parameter :: sz_sphere = szs_sphere(Mord)
    integer , parameter :: sz_sphere_p1 = szs_sphere(Mord+2)
 
    integer , parameter :: radius = (Mord -1)/2
    integer , parameter :: sz_cross = 2*Mord-1
 
 
    ! MOOD Parameters, leave to true; don't change
    logical, save :: DMP
    logical , parameter :: U2         = .true.
    logical , parameter :: U2_tol     = .true.
 
    ! Secondary variables , don't change
    real(16), parameter :: dx_16 = Lx_16/lf, dy_16 = Ly_16/nf
    integer , parameter :: lb = 1-ngc, le = lf + ngc, nb = 1-ngc, ne = nf + ngc
    real(PR), parameter :: dx = real(dx_16,PR), dy = real(dy_16,PR), Lx = real(Lx_16,PR), Ly = real(Ly_16,PR)
 
    real(16), parameter :: l_16 = 12.*min(dx_16,dy_16) !/ell
 
 
 end module
 