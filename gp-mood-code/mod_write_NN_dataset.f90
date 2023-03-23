module mod_write_NN_dataset
    use constants
    use parameters
    use GP_init

implicit none

contains

subroutine write_NN_datatset(Uin, CellGPO)

    real(PR),  intent(in), dimension(4,lb:le, nb:ne) :: Uin
    integer ,  intent(in), dimension(lb:le, nb:ne)   :: CellGPO
    real(PR), dimension(4,sz_sphere_p1) :: q_sp
    real(4) , dimension(57) :: formatted_input=-66666666

    integer :: n,l,i,j
    character(len=7) :: test_case

    logical :: cst, exist

    character(len=3) :: CFL_string

    test_case = file(18:18+7)
    write(CFL_string, '(f3.1)') CFL

    inquire(file="TD_"//test_case//"_CFL_"//CFL_string//".txt", exist=exist)
    if (exist) then
      open(10, file="TD_"//test_case//"_CFL_"//CFL_string//".txt", status="old", position="append", action="write")
    else
      open(10, file="TD_"//test_case//"_CFL_"//CFL_string//".txt", status="new", action="write")
    end if

    do n = nb+ngc, ne-ngc
        do l = lb+ngc,le-ngc

            do j = 1, sz_sphere_p1 ! Getting the whole dependancy domain of the cell l,n that is the R'=R+1 stencil
                q_sp(:,j) = Uin(:,l+ixiy_sp1(mord+2,j,1),n+ixiy_sp1(mord+2,j,2))
            end do

            cst=.true.

            call format_input(q_sp, cst, formatted_input)

            if ((cst .eqv. .true.).and.(CellGPO(l,n)==1)) then 
                print*, 'weird'
            end if

            if (cst .eqv. .false.) then
                write(10,"(57(e12.5,' '), i3)") formatted_input(:), (CellGPO(l,n)-1)/2
            end if

        end do 
    end do

    close(10)
end subroutine write_NN_datatset


subroutine format_input(q_sp, cst, formatted_input)

    real(PR), intent(inout), dimension(4,sz_sphere_p1) :: q_sp
    logical, intent(inout) :: cst
    real(4), intent(inout), dimension(57) :: formatted_input
    real(PR) :: max, min
    integer :: j, var
    real(PR), dimension(4) :: F

    do var =1, 4
            
        max = maxval(q_sp(var,:))
        min = minval(q_sp(var,:))

        if (max-min < 1e-10) then 
            F(var) = sign(1.0,min)
            do j = 1, sz_sphere_p1
                q_sp(var,j) = 1.0
            end do
        else 
            cst=.false.
            F(var) = sign(1.0,min)*(max-min)
            do j = 1, sz_sphere_p1
                q_sp(var,j)= (q_sp(var,j)- 0.5*(min+max))*(2/(max-min))
            end do
        end if

    end do

    do j = 1, sz_sphere_p1
        formatted_input(4*(j-1)+1:4*j) = real(q_sp(:,j),kind=4)
    end do
    formatted_input(53:56) = real(F(:),kind=4)
    formatted_input(57) = real(CFL,kind=4)
    
end subroutine format_input

end module mod_write_NN_dataset
