module mesh_init

   use parameters
   use global_variables

   implicit none

contains

   subroutine init_mesh()

      integer :: l, n

      do n = nb, ne
         mesh_y(n) = n * dy - dy/2
      end do

      do l = lb, le
         mesh_x(l) = l * dx - dx/2
      end do


   end subroutine init_mesh
end module mesh_init
