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

      CHARACTER(LEN=10) :: lf_char, nf_char

      WRITE(lf_char, '(I10)') lf
      WRITE(nf_char, '(I10)') nf

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
         method_char="NN_GP_MOOD_"//trim(adjustl(NN_filename))
      else if (method == FOG) then 
         method_char="FOG"
      else 
         print*, "Error, add method to method_char list in output.f90"
         stop
      end if


      file = 'output_'//trim(adjustl(problem_char))//"_"//trim(adjustl(method_char))//"_CFL_"//trim(adjustl(CFL_char))//"_"//trim(adjustl(lf_char))//"_"//trim(adjustl(nf_char))
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

      ! NN_filename
      dims=[len(trim(adjustl(NN_filename))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "NN_filename", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(NN_filename)), dims, status)
      call h5dclose_f(dataset_id, status)

      !lf and nf
      dims=[1,1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "lf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, lf, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "nf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, nf, dims, status)
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
      call h5fcreate_f('diagnostic_'//trim(adjustl(file))//'.h5', H5F_ACC_TRUNC_F, file_id, status)
      
      ! Create dataspace for datasets
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

      ! NN_filename
      dims=[len(trim(adjustl(NN_filename))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "NN_filename", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(NN_filename)), dims, status)
      call h5dclose_f(dataset_id, status)

      !lf and nf
      dims=[1,1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "lf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, lf, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "nf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, nf, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "count_steps_NN_produced_NAN", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, count_steps_NN_produced_NAN, dims, status)
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

   subroutine write_NN_dataset_()


      integer(hid_t) :: file_id, dataspace_id, dataset_id
      integer(hsize_t), dimension(2) :: dims

      integer :: status, size

      size=0

      if (n_overwrite>=1) then 
         size=dataset_size
      else
         do while ((labels(size,1) > -665))
            size=size+1
         end do
         size=size-1
      end if

      print*,size
      ! Create a new HDF5 file
      call h5open_f(status)
      call h5fcreate_f('dataset_'//trim(adjustl(file))//'.h5', H5F_ACC_TRUNC_F, file_id, status)

       ! Create dataspace for labels
      dims = [2,size]
      call h5screate_simple_f(2, dims, dataspace_id, status)
      !write and close labels
      call h5dcreate_f(file_id, "labels", H5T_NATIVE_REAL, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, transpose(labels(1:size,1:2)), dims, status)
      call h5dclose_f(dataset_id, status)

      ! Create dataspace for inputs
      dims = [L,size]
      call h5screate_simple_f(2, dims, dataspace_id, status)
      !write and close inputs
      call h5dcreate_f(file_id, "inputs", H5T_NATIVE_REAL, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_REAL, transpose(inputs(1:size,1:L)), dims, status)
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

      ! NN_filename
      dims=[len(trim(adjustl(NN_filename))),1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "NN_filename", H5T_C_S1, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_C_S1, trim(adjustl(NN_filename)), dims, status)
      call h5dclose_f(dataset_id, status)

      !lf and nf n_overwrtie, NR0, NR1
      dims=[1,1]
      call h5screate_simple_f(1, dims, dataspace_id, status)
      call h5dcreate_f(file_id, "lf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, lf, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "nf", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, nf, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "n_overwrite", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, n_overwrite, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "NR0", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, NR0, dims, status)
      call h5dclose_f(dataset_id, status)
      call h5dcreate_f(file_id, "NR1", H5T_NATIVE_INTEGER, dataspace_id, dataset_id, status)
      call h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, NR1, dims, status)
      call h5dclose_f(dataset_id, status)

      ! Close resources
      call h5sclose_f(dataspace_id, status)
      call h5fclose_f(file_id, status)
      call h5close_f(status)

      print*,''
      print*,'======================================================================'
      print*,'   A new dataset file has been written'
      print*,'   Output directory:', 'dataset_'//trim(adjustl(file))//'_'//'.h5'
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
