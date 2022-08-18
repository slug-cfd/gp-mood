### NOTE -- DL: IT'S ALWAYS SAFE TO DO (8/18/2021)
### >> MAKE CLEAN FIRST AND THEN MAKE (E.G., MAKE CLEAN && MAKE)

## NOTE -- DL: (8/19/21): added a new command (https://stackoverflow.com/questions/816370/how-do-you-force-a-makefile-to-rebuild-a-target)
### >>> make new
### this will clean and make

FC = gfortran

## Choix des options de compilation ##
#FFLAGS = -O0 -march=native
#FFLAGS = -fdefault-real-8  -ffree-line-length-none -g -O0 -fbacktrace -fcheck=all -Wuninitialized -finit-real=snan -ffpe-trap="zero,overflow,invalid" -Wall -Wextra -pg
#FFLAGS = -fdefault-real-8  -ffree-line-length-none -g -O0 -fbacktrace -fcheck=all -Wuninitialized  -Wall

#FFLAGS = -fdefault-real-8  -ffree-line-length-none -O3 -Wall -Wextra -fbacktrace -fcheck=all -Wuninitialized
FFLAGS = -fdefault-real-8  -ffree-line-length-none -O3
## Ensemble des fichiers à supprimer lors du make clean ##
OBJS = *.o *.mod *~
.PHONY: clean

## Nom de l'exécutable##
EXEC = gp_mood.exe

#OUTPUTLOG = ./test_gp_mach100.log
#OUTPUTLOG = ./plotter/Isentropic/isen_GP3_2quad_RK3_cfl0.8_HLLC_Lx20_100_ell_1.0_cross.log
#OUTPUTLOG = ./plotter/Isentropic/pol-mood3_2quad_RK3_cfl0p8_HLLC_Lx20_050.log
#OUTPUTLOG = ./plotter/ShuOsher/shuosher1d_GP3rd_1quad_RK3_nodtRed_cfl0.8_HLLC_256.log
#OUTPUTLOG = ./plotter/Mach800/dblmach800-800_GP7_4quad_RK3_cfl0p8_HLL_600.log
#OUTPUTLOG = ./plotter/Mach800/mach100_GP7_4quad_RK3_cfl0p8_HLL_600x600.log
#OUTPUTLOG = ./plotter/2DRP_C3/2drp_pol-mood3_2quad_RK3_noDtRed_cfl0p8_HLLC_400_noCSD.log
OUTPUTLOG = ./plotter/2DRP_C3/2drp_gp5_3quad_RK4_noDtRed_cfl0p8_HLLC_400_noCSD.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP7th_4quad_RK3_nodtRed_cfl0.8_HLLC_400_Ca0p1ByDx.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP5th_3quad_RK3_nodtRed_cfl0.8_HLLC_400_Mach0p1.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP3rd_2quad_RK3_nodtRed_cfl0.8_HLLC_400_Ca0p1.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP3_2quad_RK3_cfl0.8_HLLC_400_noA_PrioriCheck.log
#OUTPUTLOG = ./plotter/Implosion/implosion_polymood3_2quad_RK3_cfl0p8_HLLC_400.log
#OUTPUTLOG = ./plotter/Implosion/implosion_GP7th_4quad_RK3_nodtRed_cfl0p8_HLLC_400.log
#OUTPUTLOG = ./plotter/RMI/rmi_gp3_2quad_RK3_nodtRed_cfl0.8_HLLC_1536x256.log
#OUTPUTLOG = ./plotter/Sedov/sedov_GP3rd_2quad_RK3_cfl0p8_HLLC_256.log
#OUTPUTLOG = ./plotter/Sedov/sedov_pol1_1quad_rk3_noDtRed_cfl0.5_hllc_128_CSD.log
#OUTPUTLOG = ./plotter/DMR/dmr_gp3_2quad_RK3_nodtRed_cfl0p8_800x200.log
#OUTPUTLOG = ./plotter/DMR/dmr_pol-mood3_2quad_RK3_nodtRed_cfl0p8_800x200.log
#OUTPUTLOG = ./plotter/Sedov/sedov_gp5th_2quad_rk3_noDtRed_cfl0.95_hllc_256_CFL-TESTING.log
#OUTPUTLOG = ./plotter/Sedov/sedov_GP7th_4quad_RK4_cfl0p8_HLLC_256.log
#OUTPUTLOG = ./plotter/Sod/sodx2d_gp3_2quad_RK3_cfl0p8_hllc_256x256.log

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
	time ./$(EXEC) #> $(OUTPUTLOG) & ## Commande exécutant automatiquement le programme ##
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

## run "make new" instead of "make"
new: clean
	$(MAKE)
