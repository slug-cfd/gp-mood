module GP_init
   use constants
   use linalg
   use global_variables
   use parameters

   implicit none

contains


   !! DL -- this function variably changes the proper kernel
   !!       based on the order (ord) on each cell
   function stcl_sz(ord)result(sz) !stencil size
      integer, intent(in) :: ord
      integer             ::  sz

      if (ord == 1 ) sz = 1
      if (ord == 3 ) sz = 5
      if (ord == 5 ) sz = 13
      if (ord == 7 ) sz = 25

   end function


   subroutine GP_presim_sphere()
      real(16), dimension(Mord, sz_sphere,sz_sphere)   :: COV_16, inv_COV_16
      real(16), dimension(Mord, sz_sphere,sz_sphere)   :: Id
      real(16), dimension(      sz_sphere,sz_sphere)   :: dkh_x, dkh_y

      real(16), dimension(Mord ,sz_sphere, iL:iB, ngp) :: T_pred_16
      real(16), dimension(      sz_sphere, iL:iB, ngp) :: dks_x, dks_y

      real(16), dimension(Mord ,sz_sphere)             :: GP_d2x_16,  GP_d2y_16

      integer   :: k, h, i, j
      integer   :: ord, npt

      real(16), dimension(5,5)  :: gauss_pt =0.

      !============ Instructions ===========!

      !We initialize de gaussian points and weights

      gauss_weight(1,1) = 1._16
      gauss_pt    (1,1) = 0._16

      gauss_weight(2,1) = 0.5_16
      gauss_weight(2,2) = 0.5_16

      gauss_pt    (2,1) = -0.5_16/sq3
      gauss_pt    (2,2) = +0.5_16/sq3

      gauss_weight(3,1) = 0.5*5./9
      gauss_weight(3,2) = 0.5*8./9
      gauss_weight(3,3) = 0.5*5./9

      gauss_pt    (3,1) = -0.5_16*sq35
      gauss_pt    (3,2) = 0._16
      gauss_pt    (3,3) = +0.5_16*sq35

      gauss_weight(4,1) = 0.5*(1./36)*(18. - sq30)
      gauss_weight(4,2) = 0.5*(1./36)*(18. + sq30)
      gauss_weight(4,3) = 0.5*(1./36)*(18. + sq30)
      gauss_weight(4,4) = 0.5*(1./36)*(18. - sq30)

      gauss_pt    (4,1) = -0.5*sqrt(3./7 + (2./7)*sq65)
      gauss_pt    (4,2) = -0.5*sqrt(3./7 - (2./7)*sq65)
      gauss_pt    (4,3) = +0.5*sqrt(3./7 - (2./7)*sq65)
      gauss_pt    (4,4) = +0.5*sqrt(3./7 + (2./7)*sq65)

      gauss_weight(5,1) = 0.5*(322. - 13*sq70)/900
      gauss_weight(5,2) = 0.5*(322. + 13*sq70)/900
      gauss_weight(5,3) = 0.5*128./225
      gauss_weight(5,4) = 0.5*(322. + 13*sq70)/900
      gauss_weight(5,5) = 0.5*(322. - 13*sq70)/900
   
   
      gauss_pt    (5,1) = -0.5*third*sqrt(5. + 2*sq107)
      gauss_pt    (5,2) = -0.5*third*sqrt(5. - 2*sq107)
      gauss_pt    (5,3) =  0.
      gauss_pt    (5,4) = +0.5*third*sqrt(5. - 2*sq107)
      gauss_pt    (5,5) = +0.5*third*sqrt(5. + 2*sq107)


      COV_16      = 0.
      inv_COV_16  = 0.
      ID          = 0.
      T_pred_16   = 0.

      ! We associate to each component of the input vector q the
      ! corresponding distances for a sphere shaped stencil
      do ord = 1, Mord,2

         npt   = stcl_sz(ord)

         ixiy_sp(ord,1, dir_x) = 0
         ixiy_sp(ord,1, dir_y) = 0

         if (ord >= 3) then
            ixiy_sp(ord,2, dir_x) = -1
            ixiy_sp(ord,2, dir_y) =  0

            ixiy_sp(ord,3, dir_x) = 0
            ixiy_sp(ord,3, dir_y) = 1

            ixiy_sp(ord,4, dir_x) = 1
            ixiy_sp(ord,4, dir_y) = 0

            ixiy_sp(ord,5, dir_x) = 0
            ixiy_sp(ord,5, dir_y) = -1
         end if

         if (ord >= 5) then
            ixiy_sp(ord,6, dir_x) = -1
            ixiy_sp(ord,6, dir_y) = -1

            ixiy_sp(ord,7, dir_x) = -1
            ixiy_sp(ord,7, dir_y) =  1

            ixiy_sp(ord,8, dir_x) = 1
            ixiy_sp(ord,8, dir_y) = 1

            ixiy_sp(ord,9, dir_x) = 1
            ixiy_sp(ord,9, dir_y) = -1

            ixiy_sp(ord,10, dir_x) = 0
            ixiy_sp(ord,10, dir_y) = -2

            ixiy_sp(ord,11, dir_x) = -2
            ixiy_sp(ord,11, dir_y) = 0

            ixiy_sp(ord,12, dir_x) = 0
            ixiy_sp(ord,12, dir_y) = 2

            ixiy_sp(ord,13, dir_x) = 2
            ixiy_sp(ord,13, dir_y) = 0
         end if

         if (ord >= 7) then
            ixiy_sp(ord,14, dir_x) = -3
            ixiy_sp(ord,14, dir_y) = 0

            ixiy_sp(ord,15, dir_x) = -2
            ixiy_sp(ord,15, dir_y) = 1

            ixiy_sp(ord,16, dir_x) = -1
            ixiy_sp(ord,16, dir_y) = 2

            ixiy_sp(ord,17, dir_x) = 0
            ixiy_sp(ord,17, dir_y) = 3

            ixiy_sp(ord,18, dir_x) = 1
            ixiy_sp(ord,18, dir_y) = 2

            ixiy_sp(ord,19, dir_x) = 2
            ixiy_sp(ord,19, dir_y) = 1

            ixiy_sp(ord,20, dir_x) = 3
            ixiy_sp(ord,20, dir_y) = 0

            ixiy_sp(ord,21, dir_x) = 2
            ixiy_sp(ord,21, dir_y) = -1

            ixiy_sp(ord,22, dir_x) = 1
            ixiy_sp(ord,22, dir_y) = -2

            ixiy_sp(ord,23, dir_x) = 0
            ixiy_sp(ord,23, dir_y) = -3

            ixiy_sp(ord,24, dir_x) = -1
            ixiy_sp(ord,24, dir_y) = -2

            ixiy_sp(ord,25, dir_x) = -2
            ixiy_sp(ord,25, dir_y) = -1

         end if


         ! Computation of the cov matrix
         do k = 1, npt
            do h = 1, npt
               dkh_x(k,h)      = real(ixiy_sp(ord,k, dir_x)-ixiy_sp(ord,h, dir_x),16) !DL -- 16 for a quadruple precision
               dkh_y(k,h)      = real(ixiy_sp(ord,k, dir_y)-ixiy_sp(ord,h, dir_y),16)
               COV_16(ord,k,h) = intg_kernel_SE(dkh_x(k,h), dkh_y(k,h)) ! DL -- Eqn (11)
            end do
         end do

         Id = 0.

         ! We compute the distances between the sentcil's centers
         ! to the gaussian quadratures point on the edges

         do k = 1,npt
            Id(ord,k,k) = 1._16

            do h = 1, ngp
               ! DL -- a (x,y) pair for the one on the left edge
               dks_x(k,iL,h) = real(ixiy_sp(ord,k, dir_x),16) - (-1._16/2)
               dks_y(k,iL,h) = real(ixiy_sp(ord,k, dir_y),16) - gauss_pt(ngp,h)

               ! DL -- a (x,y) pair for the one on the top edge
               dks_x(k,iT,h) = real(ixiy_sp(ord,k, dir_x),16) - gauss_pt(ngp,h)
               dks_y(k,iT,h) = real(ixiy_sp(ord,k, dir_y),16) - (+1._16/2)

               ! DL -- a (x,y) pair for the one on the right edge
               dks_x(k,iR,h) = real(ixiy_sp(ord,k, dir_x),16) - (+1._16/2)
               dks_y(k,iR,h) = real(ixiy_sp(ord,k, dir_y),16) - gauss_pt(ngp,h)

               ! DL -- a (x,y) pair for the one on the bottom edge
               dks_x(k,iB,h) = real(ixiy_sp(ord,k, dir_x),16) - gauss_pt(ngp,h)
               dks_y(k,iB,h) = real(ixiy_sp(ord,k, dir_y),16) - (-1._16/2)
            end do



            !We compute the pred vector

            do i = iL,iB

               do j = 1, ngp
                  T_pred_16(ord,k,i,j) =                        Tpred_1D_fun(dks_x(k,i,j),dx_16)
                  T_pred_16(ord,k,i,j) = T_pred_16(ord,k,i,j) * Tpred_1D_fun(dks_y(k,i,j),dy_16)
               end do

            end do

            ! Cimputation of the GP der coeff, at the center of the cells

            ! x 2nd derivative:
            GP_d2x_16(ord,k) = Tpred_d2x_1D_fun(dkh_x(k,1),dx_16)

            ! The remaining constant factor not affected by the derivative

            GP_d2x_16(ord,k) =  GP_d2x_16(ord,k)*Tpred_1D_fun(dkh_y(k,1), dy_16)

            ! y 2nd derivative:
            GP_d2y_16(ord,k) = Tpred_d2x_1D_fun(dkh_y(k,1),dy_16)

            ! The remaining constant factor not affected by the derivative

            GP_d2y_16(ord,k) =  GP_d2y_16(ord,k)*Tpred_1D_fun(dkh_x(k,1), dx_16)

         end do

         ! We invert the cov matrices

         inv_COV_16(ord, 1:npt, 1:npt) = invChol( COV_16(ord, 1:npt, 1:npt) , npt )

         print*, ' Error in the inversion of the main kernel matrix of order', ord,"=", real( maxval (abs( ( matmul(COV_16(ord,:,:),inv_COV_16(ord,:,:)) - Id(ord,:,:)) )) ,4) , 'must be close to 0'
         error_inversion = real( maxval (abs( ( matmul(COV_16(ord,:,:),inv_COV_16(ord,:,:)) - Id(ord,:,:)) )) ,8)

         ! Computation of Zt from Tpred
         do i = iL,iB
            do j = 1, ngp

               zT_sp(ord, 1:npt, i,j) = real( matmul (T_pred_16(ord, 1:npt, i,j),inv_COV_16(ord, 1:npt,1:npt)), PR)

               !  print*,'err COV zt= T', maxval(abs(matmul(COV_16(ord, 1:npt, 1:npt),   zT_sp(ord, 1:npt, i,j)) -T_pred_16(ord, 1:npt, i,j))), 'i=',i,'j=',j

               !Normalisation
               zT_sp(ord, 1:npt, i,j) = zT_sp(ord, 1:npt, i,j)/sum(zT_sp(ord, 1:npt, i,j))

            end do
         end do

         GP_d2x_sp(ord, 1:npt) = real( matmul (GP_d2x_16(ord, 1:npt),inv_COV_16(ord, 1:npt,1:npt)), PR)
         GP_d2y_sp(ord, 1:npt) = real( matmul (GP_d2y_16(ord, 1:npt),inv_COV_16(ord, 1:npt,1:npt)), PR)

      end do

      do ord = 1, Mord+2,2

         ixiy_sp1(ord,1, dir_x) = 0
         ixiy_sp1(ord,1, dir_y) = 0

         if (ord >= 3) then
            ixiy_sp1(ord,2, dir_x) = -1
            ixiy_sp1(ord,2, dir_y) =  0

            ixiy_sp1(ord,3, dir_x) = 0
            ixiy_sp1(ord,3, dir_y) = 1

            ixiy_sp1(ord,4, dir_x) = 1
            ixiy_sp1(ord,4, dir_y) = 0

            ixiy_sp1(ord,5, dir_x) = 0
            ixiy_sp1(ord,5, dir_y) = -1
         end if

         if (ord >= 5) then
            ixiy_sp1(ord,6, dir_x) = -1
            ixiy_sp1(ord,6, dir_y) = -1

            ixiy_sp1(ord,7, dir_x) = -1
            ixiy_sp1(ord,7, dir_y) =  1

            ixiy_sp1(ord,8, dir_x) = 1
            ixiy_sp1(ord,8, dir_y) = 1

            ixiy_sp1(ord,9, dir_x) = 1
            ixiy_sp1(ord,9, dir_y) = -1

            ixiy_sp1(ord,10, dir_x) = 0
            ixiy_sp1(ord,10, dir_y) = -2

            ixiy_sp1(ord,11, dir_x) = -2
            ixiy_sp1(ord,11, dir_y) = 0

            ixiy_sp1(ord,12, dir_x) = 0
            ixiy_sp1(ord,12, dir_y) = 2

            ixiy_sp1(ord,13, dir_x) = 2
            ixiy_sp1(ord,13, dir_y) = 0
         end if

         if (ord >= 7) then
            ixiy_sp1(ord,14, dir_x) = -3
            ixiy_sp1(ord,14, dir_y) = 0

            ixiy_sp1(ord,15, dir_x) = -2
            ixiy_sp1(ord,15, dir_y) = 1

            ixiy_sp1(ord,16, dir_x) = -1
            ixiy_sp1(ord,16, dir_y) = 2

            ixiy_sp1(ord,17, dir_x) = 0
            ixiy_sp1(ord,17, dir_y) = 3

            ixiy_sp1(ord,18, dir_x) = 1
            ixiy_sp1(ord,18, dir_y) = 2

            ixiy_sp1(ord,19, dir_x) = 2
            ixiy_sp1(ord,19, dir_y) = 1

            ixiy_sp1(ord,20, dir_x) = 3
            ixiy_sp1(ord,20, dir_y) = 0

            ixiy_sp1(ord,21, dir_x) = 2
            ixiy_sp1(ord,21, dir_y) = -1

            ixiy_sp1(ord,22, dir_x) = 1
            ixiy_sp1(ord,22, dir_y) = -2

            ixiy_sp1(ord,23, dir_x) = 0
            ixiy_sp1(ord,23, dir_y) = -3

            ixiy_sp1(ord,24, dir_x) = -1
            ixiy_sp1(ord,24, dir_y) = -2

            ixiy_sp1(ord,25, dir_x) = -2
            ixiy_sp1(ord,25, dir_y) = -1

         end if
      end do

   end subroutine

   subroutine pol_presim()
      !Fill de polynomial vf extrapolation

      ! Third order
      !integer :: kM=1, kCloseNrm=2, kCloseTrans=3, kFarTrans = 4, kFarNrm = 5
      Pol_zT_o3(1, iL, 1) = o3_coef(kM) !M
      Pol_zT_o3(2, iL, 1) = o3_coef(kCloseNrm)  !L
      Pol_zT_o3(3 ,iL, 1) = o3_coef(kFarTrans)! T
      Pol_zT_o3(4 ,iL, 1) = o3_coef(kFarNrm)! R
      Pol_zT_o3(5 ,iL, 1) = o3_coef(kCloseTrans)! B

      Pol_zT_o3(1, iL, 2) = o3_coef(kM) !M
      Pol_zT_o3(2, iL, 2) = o3_coef(kCloseNrm)  !L
      Pol_zT_o3(3 ,iL, 2) = o3_coef(kCloseTrans)! T
      Pol_zT_o3(4 ,iL, 2) = o3_coef(kFarNrm)! R
      Pol_zT_o3(5 ,iL, 2) = o3_coef(kFarTrans)! B

      Pol_zT_o3(1, iR, 1) = o3_coef(kM) !M
      Pol_zT_o3(2, iR, 1) = o3_coef(kFarNrm)  !L
      Pol_zT_o3(3 ,iR, 1) = o3_coef(kFarTrans)! T
      Pol_zT_o3(4 ,iR, 1) = o3_coef(kCloseNrm)! R
      Pol_zT_o3(5 ,iR, 1) = o3_coef(kCloseTrans)! B

      Pol_zT_o3(1, iR, 2) = o3_coef(kM) !M
      Pol_zT_o3(2, iR, 2) = o3_coef(kFarNrm)  !L
      Pol_zT_o3(3 ,iR, 2) = o3_coef(kCloseTrans)! T
      Pol_zT_o3(4 ,iR, 2) = o3_coef(kCloseNrm)! R
      Pol_zT_o3(5 ,iR, 2) = o3_coef(kFarTrans)! B

      Pol_zT_o3(1, iT, 1) = o3_coef(kM) !M
      Pol_zT_o3(2, iT, 1) = o3_coef(kCloseTrans)  !L
      Pol_zT_o3(3 ,iT, 1) = o3_coef(kCloseNrm)! T
      Pol_zT_o3(4 ,iT, 1) = o3_coef(kFarTrans)! R
      Pol_zT_o3(5 ,iT, 1) = o3_coef(kFarNrm)! B

      Pol_zT_o3(1, iT, 2) = o3_coef(kM) !M
      Pol_zT_o3(2, iT, 2) = o3_coef(kFarTrans)  !L
      Pol_zT_o3(3 ,iT, 2) = o3_coef(kCloseNrm)! T
      Pol_zT_o3(4 ,iT, 2) = o3_coef(kCloseTrans)! R
      Pol_zT_o3(5 ,iT, 2) = o3_coef(kFarNrm)! B

      Pol_zT_o3(1, iB, 1) = o3_coef(kM) !M
      Pol_zT_o3(2, iB, 1) = o3_coef(kCloseTrans)  !L
      Pol_zT_o3(3 ,iB, 1) = o3_coef(kFarNrm)! T
      Pol_zT_o3(4 ,iB, 1) = o3_coef(kFarTrans)! R
      Pol_zT_o3(5 ,iB, 1) = o3_coef(kCloseNrm)! B

      Pol_zT_o3(1, iB, 2) = o3_coef(kM) !M
      Pol_zT_o3(2, iB, 2) = o3_coef(kFarTrans)  !L
      Pol_zT_o3(3 ,iB, 2) = o3_coef(kFarNrm)! T
      Pol_zT_o3(4 ,iB, 2) = o3_coef(kCloseTrans)! R
      Pol_zT_o3(5 ,iB, 2) = o3_coef(kCloseNrm)! B


      pol_zt_o5 = 0.

      pol_zT_o5(:,iT,1) = (/1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/)
      pol_zT_o5(:,iB,1) = (/1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/)
      pol_zT_o5(:,iL,1) = (1./60)*(/47.0, 27.0, 0.0, -13.0, 0.0, -3.0, 0.0, 2.0, 0.0/)
      pol_zT_o5(:,iR,1) = (1./60)*(/47.0, -13.0, 0.0,  27.0, 0.0,  2.0, 0.0, -3.0, 0.0/)


      pol_zT_o5(:,iT,2) = (/1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/)
      pol_zT_o5(:,iB,2) = (/1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/)
      pol_zT_o5(:,iL,2) = (1./60)*(/47.0, 27.0, 0.0, -13.0, 0.0, -3.0, 0.0, 2.0, 0.0/)
      pol_zT_o5(:,iR,2) = (1./60)*(/47.0, -13.0, 0.0,  27.0, 0.0,  2.0, 0.0, -3.0, 0.0/)




   end subroutine

   function Tpred_d2x_1D_fun(delta,h)result(r)

      real(16), intent(in) :: delta, h
      real(16)             :: r, expp, expm, fm, fp

      fm   = delta - 1./2
      fp   = delta + 1./2

      expm = -(fm**2)/(2*(l_16/h)**2)
      expp = -(fp**2)/(2*(l_16/h)**2)

      r = (1./l_16**2)*(fm*exp(expm) - fp*exp(expp))
   end function


   function Tpred_1D_fun(delta,h)result(r)

      real(16), intent(in) :: delta, h
      real(16)             :: r, expp, expm

      expp = (delta + 1._16/2)/( sqrt(2._16)*(l_16/h) )
      expm = (delta - 1._16/2)/( sqrt(2._16)*(l_16/h) )

      r =   sqrt(pi_16/2) * (l_16/h) * ( ERF(expp) - ERF(expm) )
   end function



   function intg_kernel_SE(ddx, ddy)result(r)
      !=========== Description ===========!

      !============= Input(s) ==============!
      real(16), intent(in):: ddx, ddy


      !============= Output(s) =============!
      real(16)            :: r

      !========== Local variables ==========!
      real(16)            :: a11, a12, a21, a22, a31, a32, lod

      !============ Instructions ===========!

      lod = l_16 / dx_16

      a11 = ( (ddx+1._16) / (sqrt(2._16)*lod) )
      a12 = ( (ddx-1._16) / (sqrt(2._16)*lod) )

      a21 = - ( ( (ddx+1._16)**2) / (2._16*lod**2) )
      a22 = - ( ( (ddx-1._16)**2) / (2._16*lod**2) )

      a31 =   ddx     /(sqrt(2._16)*lod    )
      a32 = -(ddx**2) /(     2._16 *lod**2 )

      !! DL -- eqn (11) in x-direction
      r =    (sqrt(pi_16)*(lod)**2)* &
         (a11 * ERF(a11) + a12 * ERF(a12)  &
         + (ospi)   * (       EXP(a21) +       EXP(a22)) &
         -    2._16 * ( a31 * ERF(a31) + ospi* EXP(a32)))

      lod = l_16 / dy_16


      a11 = ( (ddy+1._16) / (sqrt(2._16)*lod) )
      a12 = ( (ddy-1._16) / (sqrt(2._16)*lod) )

      a21 = - ( ( (ddy+1._16)**2) / (2._16*lod**2) )
      a22 = - ( ( (ddy-1._16)**2) / (2._16*lod**2) )

      a31 =   ddy     /(sqrt(2._16)*lod    )
      a32 = -(ddy**2) /(     2._16 *lod**2 )

      !! DL -- eqn (11) in (x-direction) x (y-direction)
      r = r * (sqrt(pi_16)*(lod)**2)* &
         (a11 * ERF(a11) + a12 * ERF(a12)  &
         + (ospi)   * (       EXP(a21) +       EXP(a22)) &
         -    2._16 * ( a31 * ERF(a31) + ospi* EXP(a32)))



   end function intg_kernel_SE
end module
