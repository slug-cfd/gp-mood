module mod_Unlim_POL
   use constants
   use Parameters
   use global_variables
   use physics
   use Gp_init
   implicit none

contains

   subroutine Unlim_POL_(Uin)

      real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

      integer                         :: l, n, k, i,j
      real, dimension(4,sz_sphere) :: q



      do n = 0, nf+1
         do l = 0, lf+1


            do k = 1, sz_sphere

               q(:,k) = Uin(:,l+ixiy_sp(mord,k,1), n+ixiy_sp(mord,k,2))

            end do


            do j = 1, ngp
               do i = iL,iB

                  if ((mord > 3).or.(ngp /= 2)) then
                     print*, 'This pol'
                     stop
                  end if


                  if (mord == 3) then

                     do k = rho, ener
                        Uh(k,i,j,l,n) = dot_product( Pol_zT_o3(1:stcl_sz(mord),i,j),q(k,1:sz_sphere) )
                     end do

                  else
                     Uh(:,i,j,l,n)=Uin(:,l,n)
                  end if

               end do
            end do

         end do
      end do




   end subroutine
end module
