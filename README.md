The GP-MOOD code used to produce the results from Bourgeois, Lee 2022 as well as a data driven approach for accelerating the MOOD method


usage of the GP-MOOD code: 

gp-mood-code requires the hdf5 librairy to be installed.

to run the 2DRP3 problem:

mv gp-mood-code
cp problems/2DRP3.f90 parameters.f90
make new


### Reproducing the results ###
Running the simulation requires a working installation of fortran and the hdf5 librairy
Running the training requires:
-python3
-pytorch
-matplotlib
-h5py
-numpy
-multiprocessing

## a given problem in {sedov, shu_osher, 2DRP3, 2DRP4, 2DRP15}
./script_auto.sh problem 9 base 256  
results are stored in results/problem/base    

