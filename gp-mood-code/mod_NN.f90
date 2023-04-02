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