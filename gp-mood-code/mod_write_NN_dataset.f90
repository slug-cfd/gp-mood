module mod_write_NN_dataset
    use constants
    use parameters
    use GP_init

implicit none

contains

subroutine write_NN_datatset(Uin, CellGPO)

    real(PR),  intent(in), dimension(4,lb:le, nb:ne) :: Uin
    integer ,  intent(in), dimension(lb:le, nb:ne)   :: CellGPO

    real(4), dimension(4,sz_sphere_p1) :: U_loc_flattened
    real(4) , dimension(57) :: formatted_input

    character(len=7) :: test_case
    character(len=3) :: CFL_string

    logical :: cst, exist
    integer :: n,l,i,j

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
                U_loc_flattened(:,j) = real( Uin(: ,l+ixiy_sp1(mord+2, j ,1) , n+ixiy_sp1(mord+2,j,2) ), kind=4)
            end do

            call format_input(U_loc_flattened, cst, formatted_input)

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

subroutine format_input(U_loc_flattened, cst, formatted_input)

    real(4), intent(inout), dimension(4,sz_sphere_p1) :: U_loc_flattened
    real(4), intent(inout), dimension(57) :: formatted_input
    logical, intent(out) :: cst

    real(4), dimension(4) :: F

    real(4) :: max, min
    integer :: j, var

    formatted_input = -6665666

    cst=.true.

    do var =1, nbvar
            
        max = maxval(U_loc_flattened(var,:))
        min = minval(U_loc_flattened(var,:))

        if (max-min < 1e-10) then 
            F(var) = sign(real(0.0,kind=4),min)
            do j = 1, sz_sphere_p1
                U_loc_flattened(var,j) = 1.0
            end do
        else 
            cst=.false.
            F(var) = sign(real(1.0,kind=4),min)*(max-min)
            do j = 1, sz_sphere_p1
                U_loc_flattened(var,j) = (U_loc_flattened(var,j)-0.5*(max+min))*(2.0/(max-min))
            end do
        end if

    end do

    do j = 1, sz_sphere_p1
        formatted_input(nbvar*(j-1)+1:nbvar*j) = U_loc_flattened(:,j)
    end do

    formatted_input(sz_sphere_p1*nbvar+1 : sz_sphere_p1*nbvar+nbvar) = F(:)
    formatted_input(57) = real(CFL,kind=4)
    
end subroutine format_input

end module mod_write_NN_dataset
