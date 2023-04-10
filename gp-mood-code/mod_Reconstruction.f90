module reconstruction
   use parameters
   use mod_FOG
   use mod_GP_SE
   use mod_GP_MOOD
   use mod_Unlim_POL
   use mod_POL_MOOD
   implicit none

contains

   subroutine recons(Uin)

      real(PR), intent(in) , dimension(4,lb:le, nb:ne) :: Uin

      if (method == FOG    )  call FOG_    (Uin)
      if (method == GP_SE  )  call GP_SE_  (Uin)
      if (method == GP_MOOD)  call GP_MOOD_(Uin)
      if (method == POL_MOOD) call POL_MOOD_(Uin)
      if (method == Unlim_POL)call Unlim_POL_(Uin)
      if (method == NN_GP_MOOD)  call GP_MOOD_(Uin)
      if (method == eval_NN_GP_MOOD)  call GP_MOOD_(Uin)

   end subroutine

end module
