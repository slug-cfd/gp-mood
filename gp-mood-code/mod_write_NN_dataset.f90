module mod_write_NN_dataset
    use constants
    use parameters
    use GP_init

implicit none

contains

subroutine write_NN_datatset(Uin, CellGPO)

    real(PR),  intent(in), dimension(4,lb:le, nb:ne) :: Uin
    integer ,  intent(in), dimension(lb:le, nb:ne)   :: CellGPO
    real(PR), dimension(4,sz_sphere) :: q_sp

    real(kind=4) :: real_numbers(25)


    real(PR), dimension(4) :: F

    integer :: n,l,j,var,i
    real(PR) :: max, min

    logical :: cst

    open(10,file='training_data.txt', status='old', action='write', position='append')

    do n = ngc, nf+ngc-1
        do l = ngc,lf+ngc-1
         !   print*, l,n
            do j = 1, sz_sphere
                q_sp(:,j) = Uin(:,l+ixiy_sp(mord,j,1),n+ixiy_sp(mord,j,2))
              !  print*,ixiy_sp(mord,j,1),ixiy_sp(mord,j,2)
            end do

            cst=.true.

            do var =1, 4
                
                max = maxval(q_sp(var,:))
                min = minval(q_sp(var,:))

                if (max-min < 1e-13) then 
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
                print*, 'weird', q_sp(4,:)
            end if

            if (cst .eqv. .false.) then 
                write(10,"(25(e12.5), i3)") real(q_sp(:,:),kind=4), real(F(:),kind=4), real(CFL,kind=4), CellGPO(l,n)
            end if

        end do 
    end do

    close(10)



end subroutine write_NN_datatset

end module mod_write_NN_dataset
