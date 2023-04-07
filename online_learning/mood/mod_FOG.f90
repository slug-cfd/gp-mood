module  mod_FOG

   use parameters
   use global_variables

   implicit none


contains

   subroutine FOG_(Uin)

      real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin


      integer :: l, n, iface


      do n = 0, nf+1
         do l = 0, lf+1

            do iface = iL,iB
               Uh(:,iface,1,l,n) = Uin(:,l,n)
            end do

         end do
      end do


   end subroutine FOG_

end module mod_FOG
