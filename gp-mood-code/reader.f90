module reader
    use global_variables
    use constants
    use parameters
    use physics
    use IC
    use hdf5
    implicit none
 
 contains

    subroutine read()

        integer(hid_t) :: file_id, dset_id, space_id, memspace_id
        integer :: rank=2, status,l,n
        integer(hsize_t), dimension(2) :: dims=(/lf, nf/)
        real(PR), dimension(4,1:lf,1:nf) :: Uprim
    
        ! Open the HDF5 file for reading
        CALL h5open_f(status)
        CALL h5fopen_f(trim(adjustl(restart_filename)), H5F_ACC_RDONLY_F, file_id, status)
    
        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "rho", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Uprim(1,1:lf,1:nf), dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "ux", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Uprim(2,1:lf,1:nf), dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "uy", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Uprim(3,1:lf,1:nf), dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "p", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Uprim(4,1:lf,1:nf), dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        ! Metadata 
        dims=[1,1]
        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "time", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_DOUBLE, t, dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        ! Open the dataset and get its dimensions
        CALL h5dopen_f(file_id, "niter", dset_id, status)
        CALL h5dget_space_f(dset_id, space_id, status)
        ! Define the dataspace for the memory buffer
        CALL h5screate_simple_f(rank, dims, memspace_id, status)
        ! Read the dataset into the memory buffer
        CALL h5dread_f(dset_id, H5T_NATIVE_INTEGER, niter, dims, status, memspace_id, space_id)
        ! Close the dataset
        CALL h5dclose_f(dset_id, status)
        CALL h5sclose_f(space_id, status)
        CALL h5sclose_f(memspace_id, status)

        !Close the file
        CALL h5fclose_f(file_id, status)

        ! Close the HDF5 library
        CALL h5close_f(status)

        ! Convert to cons variable and store in U
        do n = 1, nf
            do l = 1,lf
                U(:,l,n) = primitive_to_conservative(Uprim(:,l,n))
            end do
        end do

        if( niter >= nmax) then 
            print*,"Error while reading: niter >= nmax"
            stop 
        end if

        if( t >= tmax) then 
            print*,"Error while reading: t >= tmax"
            stop 
        end if

        end subroutine
 

 
 end module reader
 