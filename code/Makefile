FC = gfortran

## Choix des options de compilation ##
#FFLAGS = -O0 -march=native
#FFLAGS = -fdefault-real-8  -ffree-line-length-none -g -O0 -fbacktrace -fcheck=all -Wuninitialized -finit-real=snan -ffpe-trap="zero,overflow,invalid" -Wall -Wextra -pg
#FFLAGS = -fdefault-real-8  -ffree-line-length-none -g -O0 -fbacktrace -fcheck=all -Wuninitialized  -Wall

#FFLAGS = -fdefault-real-8  -ffree-line-length-none -O3 -Wall -Wextra -fbacktrace -fcheck=all -Wuninitialized
FFLAGS = -fdefault-real-8  -ffree-line-length-none -O3
## Ensemble des fichiers à supprimer lors du make clean ##
OBJS = *.o *.mod *~


## Nom de l'exécutable##
EXEC = gp_mood.exe

OUTPUTLOG = ./plotter/Isentropic/isen_GP3rd_2quad_RK3_nodtRed_cfl0.8_HLLC_Lx20_400_ell_1.0_cross.log
#OUTPUTLOG = ./plotter/ShuOsher/shuosher1d_GP7th_1quad_RK4_nodtRed_cfl0.8_HLLC_256.log
#OUTPUTLOG = ./plotter/Mach800/DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600.log
#OUTPUTLOG = ./plotter/2DRP_C3/2drp_GP3rd_2quad_RK4_noDtRed_cfl0.8_HLLC_400.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP5th_3quad_RK3_nodtRed_cfl0.8_HLLC_400.log
#OUTPUTLOG = ./plotter/RMI/rmi_gp5th_3quad_RK3_nodtRed_cfl0.8_HLLC_2400x400.log
#OUTPUTLOG = ./plotter/Sedov/sedov_GP7th_4quad_rk4_noDtRed_cfl0.8_hllc_256.log

$(EXEC):  constants.o \
	parameters.o \
	linalg.o \
	global_variables.o \
	GP_init.o \
	mesh_init.o  \
	physics.o random.o \
	IC.o  \
	mod_HLLC.o \
	mod_HLL.o \
	mod_LLF.o \
	mod_GP_MOOD.o  \
	mod_GP_SE.o  \
	mod_FOG.o \
	mod_pol_MOOD.o \
	mod_Unlim_Pol.o \
	mod_Reconstruction.o \
	BC.o \
	mod_subroutine_MOOD.o \
	mod_FE.o \
	mod_SSP_RK2.o \
	mod_SSP_RK3.o \
	mod_SSP_RK4.o \
	time_stepper.o \
	output.o  \
	main.o ## Liste des modules puis programmes à compiler ##
	$(FC) -o  $(EXEC) $^
	./$(EXEC) > $(OUTPUTLOG) & ## Commande exécutant automatiquement le programme ##
	@echo ''
	@echo ' ================================================== '
	@echo '  Output log file is written in' $(OUTPUTLOG) 
	@echo ' ================================================== '
	@echo ''


#DL -- added a suffix rule
.SUFFIXES : 
.SUFFIXES : .f90 .o

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(OBJS)
