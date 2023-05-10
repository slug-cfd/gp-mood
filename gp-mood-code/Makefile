### NOTE -- DL: IT'S ALWAYS SAFE TO DO (8/18/2021)
### >> MAKE CLEAN FIRST AND THEN MAKE (E.G., MAKE CLEAN && MAKE)

## NOTE -- DL: (8/19/21): added a new command (https://stackoverflow.com/questions/816370/how-do-you-force-a-makefile-to-rebuild-a-target)
### >>> make new
### this will clean and make

FC = gfortran

## Choix des options de compilation ##

#FFLAGS = -fdefault-real-8  -frecursive -ffree-line-length-none -g -O0 -fbacktrace -fcheck=all -Wuninitialized -Wall -Wextra -pg
FFLAGS = -fdefault-real-8  -frecursive -ffree-line-length-none -O3 

FFLAGS += -I/usr/local/include
LDFLAGS += -L/usr/local/opt/hdf5/lib -lhdf5_fortran

## Ensemble des fichiers à supprimer lors du make clean ##
OBJS = *.o *.mod *~
.PHONY: clean

## Nom de l'exécutable##
EXEC = gp_mood.exe

OUTPUTLOG = ./log.log

$(EXEC):  constants.o \
	parameters.o \
	linalg.o \
	global_variables.o \
	GP_init.o \
	mesh_init.o  \
	physics.o \
	IC.o  \
	mod_append_NN_dataset.o  \
	mod_NN.o \
	mod_HLLC.o \
	mod_HLL.o \
	mod_LLF.o \
	mod_unsplit.o \
	mod_GP_MOOD.o  \
	mod_NN_GP_MOOD_CC.o  \
	mod_GP_SE.o  \
	mod_FOG.o \
	mod_pol_MOOD.o \
	mod_Unlim_Pol.o \
	mod_Reconstruction.o \
	BC.o \
	mod_subroutine_NN_MOOD.o \
	mod_subroutine_MOOD.o \
	mod_FE.o \
	mod_SSP_RK2.o \
	mod_SSP_RK3.o \
	mod_SSP_RK4.o \
	time_stepper.o \
	reader.o \
	output.o  \ \
	main.o ## Liste des modules puis programmes à compiler ##
	$(FC) -o $(EXEC) $^ $(LDFLAGS)
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
