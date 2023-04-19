module mod_NN_GP_MOOD_CC
   use constants
   use parameters
   use global_variables
   use physics
   use GP_init
   implicit none

contains

   subroutine NN_GP_MOOD_CC_(Uin)

      real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

      integer                     :: l, n, k, i,j
      real(PR), dimension(4,sz_sphere) :: q_sp

      logical                     :: reconstruction, done
      integer                     :: ord

      real(PR) :: prob_3 

      do n = 0, nf+1
         do l = 0, lf+1

            q_sp = 0.
            done = .false.

            do i = iL,iB

               reconstruction = .false.
               if ((i==iL).and.(DetFace_x(l-1,n))) then
                  ord = min( CellGPO(l-1,n  ), CellGPO(l,n))
                  reconstruction = .true.
               else if ((i==iR).and.(DetFace_x(l,n))) then
                  ord = min( CellGPO(l+1,n  ), CellGPO(l,n))
                  reconstruction = .true.
               else if ((i==iB).and.(DetFace_y(l,n-1))) then
                  ord = min( CellGPO(l,n -1), CellGPO(l,n))
                  reconstruction = .true.
               else if ((i==iT).and.(DetFace_y(l,n))) then
                  ord = min( CellGPO(l,n+1  ), CellGPO(l,n))
                  reconstruction = .true.
               end if

               if (reconstruction .eqv. .true.) then

                  if (done .eqv. .false.) then

                     do j = 1, sz_sphere
                        q_sp(:,j) = Uin(:,l+ixiy_sp(mord,j,1),n+ixiy_sp(mord,j,2))
                     end do

                     done = .true.

                  end if


                  !! DL -- calculate the dot product between the prediction vector z^T and the local data array, q, in 1D

                  !if ((ord==3).and.(values_NN(l,n,1)>0.1)) print*,values_NN(l,n,:), l,n
                  do j = 1, ngp
                     do k = rho, ener

                        if (ord==1) then 
                           Uh(k,i,j,l,n)=Uin(k,l,n)
                        else if( ord==3 )then 
                           Uh(k,i,j,l,n) = dot_product( zT_sp(ord,1:stcl_sz(ord),i,j),q_sp(k,1:stcl_sz(ord)))
                           Uh(k,i,j,l,n) = real(values_NN(l,n,1),8)*Uin(k,l,n) + real(values_NN(l,n,2),8)*Uh(k,i,j,l,n)
                        else
                           stop
                        end if
                        
                     end do

                  end do


               end if

            end do

         end do
      end do


   end subroutine NN_GP_MOOD_CC_

end module mod_NN_GP_MOOD_CC
