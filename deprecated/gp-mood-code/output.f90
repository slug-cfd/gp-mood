module Output
  use global_variables
  use constants
  use parameters
  use physics
  use IC
  implicit none


contains

  subroutine write_slice(dir)

    integer, intent(in) :: dir

    integer             :: l, n

    real(PR), dimension(4) :: uprim


    if (dir == dir_x) then
!!$       open(50, file = trim(adjustl(file_slice_x)), form ='formatted')
       open(50, file = trim(adjustl(file))//'slice_x.dat', form ='formatted')
       do l = 1, lf
          uprim = conservative_to_primitive(U(1:4,l,nf/2))

          write(50,*)mesh_x(l),U(1:4,l,nf/2), uprim(4)/(uprim(1)*(y-1.))

          !write(50,*)mesh_x(l)-0.5,uprim, uprim(4)/(uprim(1)*(y-1.))
          !wri!te(50,*)mesh_x(l),U(:,l,1)

       end do
       close(50)
    else if (dir == dir_y) then
       open(50, file = trim(adjustl(file))//'slice_y.dat', form ='formatted')
       do n = 1, nf
          write(50,*)mesh_y(n),U(1,lf/2,n)
       end do
       close(50)

    else if (dir == dir_xy) then
       open(50, file = trim(adjustl(file))//'slice_xy.dat', form ='formatted')
       do l = 1, min(lf,nf)
          write(50,*)mesh_x(l)*cos(Pi/4)+mesh_y(l)*sin(Pi/4),U(rho,l,l)
       end do
       close(50)

    else if (dir == dir_yx) then
       open(50, file = trim(adjustl(file))//'slice_yx.dat', form ='formatted')
       do l = 1, min(lf,nf)
          write(50,*)(mesh_x(l)-0.5)*sqrt(2.),U(rho,l,nf-l+1)
       end do
       close(50)
    else
       print*, 'dir /= x or y'
       stop
    end if
  end subroutine write_slice

  subroutine write_output(fileNumb)

    integer :: l,n

    real(PR), dimension(4,1:lf,1:nf) :: Uprim
    real(PR),dimension(4) :: v, vt
    integer, intent(IN) :: fileNumb
    character(len=50) :: fileID

    !DL -- convert file number to character
    write(fileID,910) fileNumb + 100000

910 format(i6)

    do n = 1, nf
       do l = 1,lf
          Uprim(:,l,n) = conservative_to_primitive(U(1:4,l,n))
       end do
    end do


    open(50, file = trim(adjustl(file))//'_'//trim(fileID)//'.dat', form='formatted')
    write(50,*) 'x','y','rho','ux','uy','p','ordr'
    do n = 1, nf
       do l = 1, lf
          write(50,*) mesh_x(l), mesh_y(n), Uprim(:,l,n), CellGPO(l,n)
       end do
    end do
    close(50)
    
!!$    open(50, file = trim(adjustl(file))//'rho_'//trim(fileID)//'.dat', form='formatted')
!!$    do n = 1, nf
!!$
!!$       write(50,*) Uprim(rho,1:lf,n)
!!$
!!$    end do
!!$    close(50)
!!$
!!$    open(49, file = trim(adjustl(file))//'pres_'//trim(fileID)//'.dat', form='formatted')
!!$    do n = 1, nf
!!$       write(49,*) Uprim(4,1:lf,n)
!!$
!!$    end do
!!$    close(49)
!!$
!!$    open(50, file = trim(adjustl(file))//'final_sym_'//trim(fileID)//'.dat', form='formatted')
!!$    write(50,*) 'x','y','rho','ux','uy','p'
!!$
!!$    if (nf==lf) then
!!$       do n = 1, nf
!!$          do l = 1, lf
!!$
!!$             v = conservative_to_primitive(U(1:4,l,n))
!!$             vt=conservative_to_primitive(U(1:4,n,l))
!!$             write(50,*) mesh_x(l), mesh_y(n), abs(v(:)-vt)
!!$
!!$          end do
!!$       end do
!!$    end if
!!$    close(50)
!!$
!!$    open(50, file = trim(adjustl(file))//'final_sym_2_'//trim(fileID)//'.dat', form='formatted')
!!$    write(50,*) 'x','y','rho','ux','uy','p'
!!$
!!$    if (nf==lf) then
!!$       do n = 1, nf
!!$          do l = 1, lf
!!$
!!$             v = conservative_to_primitive(U(1:4,l,n))
!!$             vt =conservative_to_primitive(U(1:4,nf-l+1,lf-n+1))
!!$             write(50,*) mesh_x(l), mesh_y(n), abs(v(:)-vt)
!!$
!!$          end do
!!$       end do
!!$    end if
!!$    close(50)

  end subroutine write_output


  subroutine error()

    real(PR) :: error_1, error_2, int,xx,yy
    real(PR), dimension(4) :: sol
    integer :: l,n

    error_1 = 0. !L_1 error
    error_2 = 0. !L_2 error (newly added for a revision)


    do n = 1, nf
       do l = 1, lf


          xx = mesh_x(l)
          yy = mesh_y(n)

          if( IC_type == isentropic_vortex) then
             sol = primitive_to_conservative((1./(dx*dy))*quadrature(mesh_x(l)-dx/2,mesh_x(l)+dx/2, mesh_y(n)-dy/2,mesh_y(n)+dy/2,0.5*Lx, 0.5*Ly, 0.))
!!$             sol = primitive_to_conservative((1./(dx*dy))*quadrature(mesh_x(l)-dx/2,mesh_x(l)+dx/2, mesh_y(n)-dy/2,mesh_y(n)+dy/2,10.,10.,20.))
             
          else if(IC_type == Lin_Gauss_xy) then
             int =     (-0.0886227*erf(-5*dx - 10*xx +5.) + 0.0886227*erf(5*dx - 10*xx + 5.))
             int = int*(-0.0886227*erf(-5*dy - 10*yy +5.) + 0.0886227*erf(5*dy - 10*yy + 5.))
             int  = (int + dx*dy)/(dx*dy)
             sol(1) = int

          else if (IC_type == Lin_Gauss_x) then

             int  = -0.0886227*erf(5. - 10.*(xx+dx/2)) + 0.0886227*erf(5. - 10.*(xx-dx/2))

             int  = (int + dx)/(dx)
             sol(1) = int


          else
             sol(1) = U(rho,l,n)
          end if


          if( dim ==2) then
             error_1 = error_1 + abs(sol(1) - U(rho,l,n))*dx*dy ! L_1 error
             !error_2 = error_2 + sqrt( ( dx*dy*(sol(1) - U(rho,l,n))**2 ) ) ! L_2 error -- this is wrong
             error_2 = error_2 + (sol(1) - U(rho,l,n))**2
          endif
          
          if( (dim ==1) .and. (n==1)) then
             error_1 = error_1 + abs(sol(1) - U(rho,l,n))*dx
          endif


       end do
    end do

    if ( (IC_type == isentropic_vortex) .or.  ( (IC_type == Lin_Gauss_xy) .or.(IC_type == Lin_Gauss_x) )) then

       print*, 'L1 error = ', error_1
       open (50, file = trim(adjustl(file))//'error_L1.dat', form='formatted',position='append')
       write(50,*)lf,nf,error_1, error_inversion
       close(50)

       if (dim == 2) then
          error_2 = dx*dy*error_2
          error_2 = sqrt(error_2)
       endif
       
       print*, 'L2 error = ', error_2
       open (60, file = trim(adjustl(file))//'error_L2.dat', form='formatted',position='append')
       write(60,*)lf,nf,error_2, error_inversion
       close(60)

    end if
  end subroutine error



end module Output
