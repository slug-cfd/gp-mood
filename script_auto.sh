#Example: ./script_auto.sh 2DRP3 9 base 256
echo $1 #Problem
echo $2 #Number of trained NN
echo $3 #Name of the folder for saving the results
echo $4 #Resolution of the problem

cd gp-mood-code
cp online_learning/$1_dataset.f90 parameters.f90
make new
cd ../neural_network
python3.9 wrapper_train_multi_cpu.py $1 $2 $4
for file in *.pt; do python3.9 save_NN_for_fortran.py $file ;done
mv *.txt ../gp-mood-code
cd ../gp-mood-code
for i in $(seq 0 $2);
do
  echo "Iteration $i"
  cp online_learning/$1_eval.f90 parameters.f90
  sed "s/NN_string/model_$1_first_10%_CEL_dropout_0.1_"$i"_5/g" parameters.f90 >parameters.tmp && mv parameters.tmp parameters.f90
  if [ "$i" -eq $2 ]; then
    make new 
  else
    make new &
    sleep 60
  fi  
done

cp problems/$1_GPMOOD.f90 parameters.f90 ; make new
cp problems/$1_FOG.f90 parameters.f90 ; make new
cp problems/$1_GPMOOD_noDMP.f90 parameters.f90 ; make new

for file in diagnostic_output* ; do python3.9 plot/plotter_diag.py $file ; done
for file in output_*           ; do python3.9 plot/plotter_hdf5.py $file rho ; done
for file in output_*           ; do python3.9 plot/plotter_hdf5.py $file ordr ; done
make clean
cd ../results
mkdir $1
mkdir $1/$3
mv ../gp-mood-code/*.txt $1/$3
mv ../gp-mood-code/*.png $1/$3
mv ../gp-mood-code/*.h5 $1/$3
mv ../neural_network/*.pt $1/$3
mv ../neural_network/*.png $1/$3

cd $1/$3
python3.9 ../../compare_diagnostic.py $1 $4
