module Output
   use global_variables
   use constants
   use parameters
   use physics
   use IC
   use hdf5
   implicit none


contains

   subroutine compute_metadata()

      write(CFL_char, "(F3.1)") CFL

      if (problem == RP_2D_3 ) then 
         problem_char='2DRP3'
      else if(problem == RP_2D_4) then 
         problem_char='2DRP4'
      else if(problem == RP_2D_6) then 
         problem_char='2DRP6'
      else if(problem == RP_2D_12) then 
         problem_char='2DRP12'
      else if(problem == RP_2D_15) then 
         problem_char='2DRP15'
      else if(problem == RP_2D_17) then 
         problem_char='2DRP17'
      else if(problem == implosion ) then 
         problem_char='implosion'
      else if(problem == sedov) then 
         problem_char='sedov'
      else if(problem == Shu_Osher_rotated ) then 
         problem_char='shu_osher'
      else if(problem == DMR) then 
         problem_char='DMR'
      else if(problem == explosion) then 
         problem_char='explosion'
      else 
         print*, "Error, add problem to problem_char list in output.f90"
         stop
      end if

      if (method == GP_MOOD) then 
         method_char="GP_MOOD"
      else if (method == POL_MOOD) then 
         method_char="POL_MOOD"
      else if (method == NN_GP_MOOD) then 
         method_char="NN_GP_MOOD"
      else if (method == FOG) then 
         method_char="FOG"
      else 
         print*, "Error, add method to method_char list in output.f90"
         stop
      end if

      file = 'output_'//trim(adjustl(problem_char))//"_"//trim(adjustl(method_char))//"_CFL_"//trim(adjustl(CFL_char))
   end subroutine
      

   subroutine write_output(fileNumb)

      integer :: l,n

      real(PR), dimension(4,1:lf,1:nf) :: Uprim
      integer, intent(IN) :: fileNumb
      character(len=50) :: fileID

      integer(hid_t) :: file_id, dataspace_id, dataset_id
      integer(hsize_t), dimension(2) :: dims


      integer :: status

      ! Compute file ID
      write(fileID,910) fileNumb + 100000

      910   format(i6)

      ! Convert to prim variable
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

      ! Write and close each dataset
      call h5dcreate_f(file_id, "rho", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(1,1:lf,1:nf), dims, status)
      call h5dclose_f(dataset_id, status)

      call h5dcreate_f(file_id, "ux", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(2,1:lf,1:nf), dims, status)
      call h5dclose_f(dataset_id, status)

      call h5dcreate_f(file_id, "uy", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(3,1:lf,1:nf), dims, status)
      call h5dclose_f(dataset_id, status)

      call h5dcreate_f(file_id, "p", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, Uprim(4,1:lf,1:nf), dims, status)
      call h5dclose_f(dataset_id, status)

      call h5dcreate_f(file_id, "ordr", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, CellGPO(1:lf, 1:nf), dims, status)
      call h5dclose_f(dataset_id, status)

      ! Write metadata

      ! CFL
      dims=[len(trim(adjustl(CFL_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "CFL", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(CFL_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! problem
      dims=[len(trim(adjustl(problem_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "problem", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(problem_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! method
      dims=[len(trim(adjustl(method_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "method", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(method_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! Close resources
      call h5fclose_f(file_id, status)
      call h5close_f(status)

      print*,''
      print*,'======================================================================'
      print*,'   A new output has been written, file number=',niter
      print*,'   Output directory:', file
      print*,'======================================================================'
      print*,''
   end subroutine 

   subroutine write_diagnostic()


      integer(hid_t) :: file_id, dataspace_id, dataset_id
      integer(hsize_t), dimension(2) :: dims

      integer :: status, size

      size=0
      do while (time(size) .ne. -666)
         size=size+1
      end do
      size=size-1

      
      ! Create a new HDF5 file
      call h5open_f(status)
      call h5fcreate_f('diagnostic_'//trim(adjustl(file))//'_'//'.h5', H5F_ACC_TRUNC_F, file_id, status)
      
      ! Create dataspace for datasets and 
      dims = [size,1]
      call h5screate_simple_f(1, dims, dataspace_id, status)

      !write and close datasets
      call h5dcreate_f(file_id, "time", H5T_NATIVE_REAL, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, time(1:size), dims, status)
      call h5dclose_f(dataset_id, status)

      call h5dcreate_f(file_id, "pct_detected_cell", H5T_NATIVE_REAL, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, pct_detected_cell(1:size), dims, status)
      call h5dclose_f(dataset_id, status)

      ! Write metadata

      ! CFL
      dims=[len(trim(adjustl(CFL_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "CFL", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(CFL_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! problem
      dims=[len(trim(adjustl(problem_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "problem", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(problem_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! method
      dims=[len(trim(adjustl(method_char))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "method", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(method_char)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! Close resources
      call h5sclose_f(dataspace_id, status)
      call h5fclose_f(file_id, status)
      
      call h5close_f(status)

      print*,''
      print*,'======================================================================'
      print*,'   A new diag file has been written'
      print*,'   Output directory:', 'diagnostic_'//trim(adjustl(file))//'_'//'.h5'
      print*,'======================================================================'
      print*,''
   end subroutine


   subroutine write_output_old(fileNumb)

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


end module Output
