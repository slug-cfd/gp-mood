module mod_NN

    use global_variables
    use mod_write_NN_dataset
    use constants
    use parameters
    use GP_init

    implicit none

contains

    subroutine compute_CellGPO_with_NN(Uin)

        real(PR), dimension(4,lb:le, nb:ne ),intent(in) :: Uin
        real(4) , dimension(57) :: x
        real(4) , dimension(4,sz_sphere_p1) :: U_loc_flattened

        integer :: l, n,j, count
        logical :: cst

        real(4), dimension(2) :: r

        count=0

        do n = 1, nf
            do l = 1, lf

                do j = 1, sz_sphere_p1 ! Getting the whole dependancy domain of the cell l,n that is the R'=R+1 stencil
                    U_loc_flattened(:,j) = real( Uin(: ,l+ixiy_sp1(mord+2, j ,1) , n+ixiy_sp1(mord+2,j,2) ), kind=4)
                end do
    
                cst=.true.
    
                call format_input(U_loc_flattened, cst, x)

                if (cst) then 
                    CellGPO(l,n)=3
                else 
                    r=forward(x)
                    if (r(1)>r(2)) then 
                        CellGPO(l,n)=1
                        count=count+1
                    else 
                        CellGPO(l,n)=3
                    end if
                end if

            end do 
        end do 

        count_detected_cell=count_detected_cell+count
    end subroutine

    subroutine load_NN(filename)

        character(len=100), intent(in) :: filename

        real(4), dimension(2) :: r
        real(4), dimension(L) :: x

        integer :: n,i,j
    
        ! Open the file for reading
        open(10, file=adjustl(filename), status='old', action='read')
    
        ! Read each line of the file and process the number
    
        n = 0
        i = 1
        j = 1
    
        do
            n=n+1
    
            if ( n <= up0) then 
    
                read(10, *) weight0(i,j)
                call update_ij(i,j, weight0)
    
            else if ((n > up0).and.( n <= up1) ) then 
    
                read(10, *) bias0(i,j)
                call update_ij(i,j, bias0)
    
            else if ((n > up1).and.( n <= up2) ) then 
    
                read(10, *) weight1(i,j)
                call update_ij(i,j, weight1)
    
            else if ((n > up2).and.( n <= up3) ) then 
    
                read(10, *) bias1(i,j)
                call update_ij(i,j, bias1)
    
            else if ((n > up3).and.( n <= up4) ) then 
    
                read(10, *) weight2(i,j)
                call update_ij(i,j, weight2)
    
            else if ((n > up4).and.( n <= up5) ) then 
                read(10, *) bias2(i,j)
                call update_ij(i,j, bias2)
            else if (n>up5) then 
                exit 
            end if
    
        end do
        ! Close the file
        close(10)
        x=1
        r=forward(x)
        print*,r
        print*,'NN LOADED SUCCESSFULLY'
    end subroutine load_NN

    subroutine sigmoid(x)

        real(4), dimension(:), intent(inout) :: x
        integer :: NL, i

        NL=size(x)

        do i =1, NL 
            x(i)=1.0 / (1.0 + exp(-x(i)))
        end do
       
    end subroutine sigmoid

    subroutine eval_NN(first)

        integer :: l,n

        character(len=7) :: test_case
        character(len=3) :: CFL_string
        logical :: exist
        logical, intent(in) :: first

        real(PR) :: error_pred
        real(PR) :: fraction_of_R0_instead_of_R1
        real(PR) :: fraction_of_R1_instead_of_R0
        real(PR) :: fraction_of_R0_instead_of_R1_relative
        real(PR) :: fraction_of_R1_instead_of_R0_relative
        
        test_case = file(18:18+7)
        write(CFL_string, '(f3.1)') CFL

        if (first) then
            NWRONG=0
            N_R0_instead_of_R1=0
            N_R1_instead_of_R0=0
            NR0_according_to_MOOD=0
            NR1_according_to_MOOD=0
        end if


        do n = 1, nf
           do l = 1, lf

              if (CellGPO_MOOD(l,n)==1) then 
                NR0_according_to_MOOD=NR0_according_to_MOOD+1
              else
                NR1_according_to_MOOD=NR1_according_to_MOOD+1
              end if

              if (CellGPO(l,n)>CellGPO_MOOD(l,n)) then 
                 N_R1_instead_of_R0=N_R1_instead_of_R0+1
                 NWRONG=NWRONG+1
              else if  (CellGPO(l,n)<CellGPO_MOOD(l,n)) then 
                 N_R0_instead_of_R1=N_R0_instead_of_R1+1
                 NWRONG=NWRONG+1
              end if
           end do 
        end do

        print*, N_R1_instead_of_R0, NR0_according_to_MOOD, N_R1_instead_of_R0*100.0/NR0_according_to_MOOD

        if (last_RK_stage) then
            print*,NWRONG
            inquire(file="eval_PROBLEM_"//test_case//"_CFL_"//CFL_string//"_MODEL_"//trim(adjustl(NN_filename(3:))), exist=exist)
            if (exist) then
                open(10, file="eval_PROBLEM_"//test_case//"_CFL_"//CFL_string//"_MODEL_"//trim(adjustl(NN_filename(3:))), status="old", position="append", action="write")
            else
                open(10, file="eval_PROBLEM_"//test_case//"_CFL_"//CFL_string//"_MODEL_"//trim(adjustl(NN_filename(3:))), status="new", action="write")
                write(10,*) test_case
                write(10,*) CFL_string
                write(10,*) trim(adjustl(NN_filename(3:)))
                write(10,*) lf
                write(10,*) nf

            end if

            error_pred=NWRONG*100.0/(3*lf*nf)
            fraction_of_R0_instead_of_R1=N_R0_instead_of_R1*100.0/(3*lf*nf)
            fraction_of_R1_instead_of_R0=N_R1_instead_of_R0*100.0/(3*lf*nf)
            fraction_of_R1_instead_of_R0_relative=N_R1_instead_of_R0*100.0/NR0_according_to_MOOD
            fraction_of_R0_instead_of_R1_relative=N_R0_instead_of_R1*100.0/NR1_according_to_MOOD

            if (NR0_according_to_MOOD==0) then 
                fraction_of_R1_instead_of_R0_relative=0 
            end if

            if (NR1_according_to_MOOD==0) then 
                fraction_of_R0_instead_of_R1_relative=0 
            end if
            write(10,*) error_pred, fraction_of_R0_instead_of_R1, fraction_of_R1_instead_of_R0, count_NN_need_posteriori_correction, fraction_of_R0_instead_of_R1_relative,  fraction_of_R1_instead_of_R0_relative, NR0_according_to_MOOD

            close(10)

        end if

  

    end subroutine

    subroutine update_ij(i, j, array)

        integer, intent(inout) :: i,j 
        real(4), dimension(:,:), intent(in) :: array
        integer :: NL, NC
        
        NL=size(array, dim=1)
        NC=size(array, dim=2)

        j=j+1
        if (j == NC+1) then 
            j=1 
            i=i+1 
            if(i == NL+1) then 
                i=1
                !print*,"End of tensor", NL, NC
            end if 
        end if 

    end subroutine update_ij

    function forward(x)result(r)

        real(4), dimension(L),intent(in) :: x
        real(4), dimension(2) :: r

        real(4), dimension(lenght) :: x0
        real(4), dimension(lenght) :: x1


        x0 = matmul(weight0, x) + bias0(:,1)
        call sigmoid(x0)
        x1 = matmul(weight1, x0) + bias1(:,1)
        call sigmoid(x1)
        r = matmul(weight2, x1) + bias2(:,1)
        
    end function forward

end module mod_NN