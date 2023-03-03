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

    real(kind=4) :: real_numbers(25)


    real(PR), dimension(4) :: F

    integer :: n,l,j,var,i
    real(PR) :: max, min

    logical :: cst

    open(10,file='training_data.txt', status='old', action='write', position='append')

    do n = nb+ngc, ne-ngc
        do l = lb+ngc,le-ngc

            do j = 1, sz_sphere_p1 ! Getting the whole dependancy domain of the cell l,n that is the R'=R+1 stencil
                q_sp(:,j) = Uin(:,l+ixiy_sp1(mord+2,j,1),n+ixiy_sp1(mord+2,j,2))
            end do

            cst=.true.

            do var =1, 4
                
                max = maxval(q_sp(var,:))
                min = minval(q_sp(var,:))

                if (max-min < 1e-14) then 
                    F(var) = sign(1.0,min)
                    do j = 1, sz_sphere
                         q_sp(var,j)= 1.0
                    end do
                else 
                    cst=.false.
                    F(var) = sign(1.0,min)*(max-min)
                    do j = 1, sz_sphere
                        q_sp(var,j)= (q_sp(var,j)- 0.5*(min+max))*(2/(max-min))
                    end do
                end if

            end do

            if ((cst .eqv. .true.).and.(CellGPO(l,n)==1)) then 
                print*, 'weird'
            end if

            write(10,"(57(e12.5), i3)") real(q_sp(:,:),kind=4), real(F(:),kind=4), real(CFL,kind=4), CellGPO(l,n)

        end do 
    end do

    close(10)



end subroutine write_NN_datatset

end module mod_write_NN_dataset
