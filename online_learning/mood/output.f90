module Output
   use global_variables
   use constants
   use parameters
   use physics
   use IC
   implicit none


contains

   subroutine write_output(fileNumb)

      integer :: l,n

      real(PR), dimension(4,1:lf,1:nf) :: Uprim
      integer, intent(IN) :: fileNumb
      character(len=50) :: fileID

      write(fileID,910) fileNumb + 100000

910   format(i6)

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

      print*,''
      print*,'======================================================================'
      print*,'   A new output has been written, file number=',niter
      print*,'   Output directory:', file
      print*,'======================================================================'
      print*,''
   end subroutine write_output

end module Output
