module Output
   use global_variables
   use constants
   use parameters
   use physics
   use IC
   use hdf5
   implicit none


contains

   subroutine write_output_no(fileNumb)

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
   end subroutine 

   subroutine write_output(fileNumb)

      integer :: l,n

      real(PR), dimension(4,1:lf,1:nf) :: Uprim
      integer, intent(IN) :: fileNumb
      character(len=50) :: fileID

      integer(hid_t) :: file_id, dataspace_id, dataset_id
      integer(hsize_t), dimension(2) :: dims

      integer :: status

      write(fileID,910) fileNumb + 100000

      910   format(i6)

      do n = 1, nf
         do l = 1,lf
            Uprim(:,l,n) = conservative_to_primitive(U(1:4,l,n))
         end do
      end do

      ! Create a new HDF5 file
      call h5open_f(status)
      
      call h5fcreate_f(trim(adjustl(file))//'_'//trim(fileID)//'.h5', H5F_ACC_TRUNC_F, file_id, status)
      ! Create dataspace for datasets
      dims = [lf, nf]
      call h5screate_simple_f(2, dims, dataspace_id, status)

      ! Create first dataset with key "rho"
      call h5dcreate_f(file_id, "rho", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(1,1:lf,1:nf), dims, status)

      call h5dcreate_f(file_id, "ux", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(2,1:lf,1:nf), dims, status)

      call h5dcreate_f(file_id, "uy", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(3,1:lf,1:nf), dims, status)

      call h5dcreate_f(file_id, "p", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(4,1:lf,1:nf), dims, status)

      call h5dcreate_f(file_id, "ordr", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, CellGPO(1:lf, 1:nf), dims, status)

      ! Close resources
      call h5dclose_f(dataset_id, status)
      call h5sclose_f(dataspace_id, status)
      call h5fclose_f(file_id, status)
      
      call h5close_f(status)

      print*,''
      print*,'======================================================================'
      print*,'   A new output has been written, file number=',niter
      print*,'   Output directory:', file
      print*,'======================================================================'
      print*,''
   end subroutine 

end module Output
