module mod_append_NN_dataset
   use constants
   use parameters
   use GP_init
   use physics

   implicit none

contains

   subroutine append_to_NN_datatset(Uin)

      real(PR),  intent(in), dimension(4,lb:le, nb:ne) :: Uin

      real(4), dimension(4,sz_sphere_p1) :: U_loc_flattened
      real(4) , dimension(L) :: formatted_input

      logical :: cst, skip_for_balanced_dataset

      real :: random_num

      integer :: n,l_,j

      do n =  1, nf
         do l_ = 1, lf

            do j = 1, sz_sphere_p1 ! Getting the whole dependancy domain of the cell l,n that is the R'=R+1 stencil
              !U_loc_flattened(:,j) = real( conservative_to_primitive( Uin(: ,l_+ixiy_sp1(mord+2, j ,1) , n+ixiy_sp1(mord+2,j,2) ) ), kind=4)
               U_loc_flattened(:,j) = real(                            Uin(: ,l_+ixiy_sp1(mord+2, j ,1) , n+ixiy_sp1(mord+2,j,2)   ), kind=4)

            end do

            call format_input(U_loc_flattened, cst, formatted_input)

            if ((cst .eqv. .true.).and.(CellGPO(l_,n)==1)) then
               print*, 'weird'
               stop
            end if

            skip_for_balanced_dataset=.false.

            if (CellGPO(l_,n)==3) then 
               call random_number(random_num)

               if (random_num>freq_R0) then 
                  skip_for_balanced_dataset=.true.
               end if

            end if

            if ((cst .eqv. .false.) .and. (skip_for_balanced_dataset .eqv. .false.)) then

               inputs(index,1:L)=formatted_input(1:L)

               if (CellGPO(l_,n)==3) then 
                  labels(index,1:2)=(/zero,one/)
                  NR1=NR1+1
               else
                  labels(index,1:2)=(/one,zero/)
                  NR0=NR0+1
               end if

               freq_R0=NR0*one/(NR0+NR1)
               
               if (index == dataset_size) then 
                  index=1
                  print*,"reached end of memory buffer. Starting overwriting ..."
                  n_overwrite=n_overwrite+1
               else
                  index=index+1
               end if

            end if

         end do
      end do

      close(10)
   end subroutine append_to_NN_datatset

   subroutine format_input(U_loc_flattened, cst, formatted_input)

      real(4), intent(inout), dimension(4,sz_sphere_p1) :: U_loc_flattened
      real(4), intent(inout), dimension(L) :: formatted_input
      logical, intent(out) :: cst

      real(4), dimension(4) :: F

      real(4) :: max, min
      integer :: j, var

      formatted_input = -666

      cst=.true.

      do var =1, nbvar

         max = maxval(U_loc_flattened(var,:))
         min = minval(U_loc_flattened(var,:))

         if (max-min < 1e-10) then
            !F(var) = 0.0
            !do j = 1, sz_sphere_p1
               !U_loc_flattened(var,j) = 1.0
            !end do
         else
            cst=.false.
            !F(var) = sign(real(1.0,kind=4),min)*(max-min)
            !do j = 1, sz_sphere_p1
              !U_loc_flattened(var,j) = (U_loc_flattened(var,j)-half*(max+min))*(two/(max-min))
            !end do
         end if

      end do

      do j = 1, sz_sphere_p1
         formatted_input(nbvar*(j-1)+1:nbvar*j) = U_loc_flattened(:,j)
      end do

      !formatted_input(sz_sphere_p1*nbvar+1 : sz_sphere_p1*nbvar+nbvar) = F(:)
      !formatted_input(sz_sphere_p1*nbvar+1 : sz_sphere_p1*nbvar+nbvar) = 0.0
      !formatted_input(L) = real(CFL,kind=4)
      !formatted_input(L) = 0.0

   end subroutine format_input

   function criterion_niter_f()result(criterion_iter)

      integer :: nstep_at_max_CFL, nstep, freq
      integer :: noutput = 50
      logical :: criterion_iter

      if (problem==RP_2D_3) then
         nstep_at_max_CFL = 600
      else if (problem==RP_2D_4) then
         nstep_at_max_CFL = 201
      else if (problem==RP_2D_6) then
         nstep_at_max_CFL = 225
      else if (problem==RP_2D_12) then
         nstep_at_max_CFL = 170
      else if (problem==RP_2D_15) then
         nstep_at_max_CFL = 128
      else if (problem==RP_2D_17) then
         nstep_at_max_CFL = 195
      else if (problem==DMR) then
         nstep_at_max_CFL = 500
      else if (problem==implosion) then
         nstep_at_max_CFL = 5460
      else if (problem==sedov) then
         nstep_at_max_CFL = 200
      else if (problem==Shu_Osher_rotated) then
         nstep_at_max_CFL = 81
      else if (problem==explosion) then
         nstep_at_max_CFL = 1500
      else if (problem==Mach800) then
         nstep_at_max_CFL = 1500
      else if (problem==RT) then
         nstep_at_max_CFL = 100
      else
         !print*,"error, add problem to problme list in mod_append_NN_dataset.f90"
         !stop
      end if

      nstep = nstep_at_max_CFL * int(0.8/CFL)
      freq = max(nstep/noutput,1) ! Avoid division by 0

      criterion_iter=(mod(niter, freq)==0)

      if ((problem == RP_2D_6)) then
         if (count_detected_cell_RK>0) then 
            criterion_iter=.true.
         end if
      end if

      if( problem == implosion) then 
         if ((t>1.0).and.(count_detected_cell_RK>0)) then 
            criterion_iter=.true.
         end if
      end if

      criterion_iter=.true. ! Forcing to write at each time step, ONLY FOR ONLINE LEARNING

   end function

end module mod_append_NN_dataset
