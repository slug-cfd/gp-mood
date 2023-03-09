import os
from NN import *
import sys
from utils import *
#Takes a txt file from gp mood and output two torch arrays correspond to R=1 an R=0
#for file in ../gp-mood-code/*.txt ; do python3.9 txt_to_torch.py "$file" ;  done
def txt_to_torch(file_name):

    print(colors.HEADER+"Converting ", file_name, " to torch arrays"+colors.ENDC)
    print("trimming the txt file ...")
    trimmed_path=dir+'trimmed_'+file_name[len(dir):]
    os.system('sort '+file_name+'| uniq > "'+trimmed_path+'"')
    print("done")

    output_path0 = 'data/'+file_name[len(dir):-4]+'_0_torch.pt'
    output_path1 = 'data/'+file_name[len(dir):-4]+'_1_torch.pt'

    print(colors.yellow+"will save data to",output_path0,"and",output_path1+colors.ENDC)

    print("enumerating ...")

    N0=0
    N1=0

    with open(trimmed_path, 'r') as file:
        for line in file:

            columns = line.split()
            columns_float = [float(column) for column in columns]

            if (columns_float[-1]==0):
                N0+=1
            else:
                N1+=1


    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))

    print(N0,"entry with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    print(colors.yellow+"gathering and writing ..."+colors.ENDC)

    with open(trimmed_path, 'r') as file:
            
            iline0=0
            iline1=0

            for line in file:
                
                columns = line.split()
                columns_float = [float(column) for column in columns]

                if (columns_float[-1]==0):
                    for k in range(L):
                        input0[iline0,k]=columns_float[k]
                    iline0+=1
                else:
                    for k in range(L):
                        input1[iline1,k]=columns_float[k]
                    iline1+=1

    torch.save(input0, output_path0)
    torch.save(input1, output_path1)
    print("end")

if (len(sys.argv)<2):
    print("error, usage: python3 txt_to_torch.py file.txt")
    sys.exit()

txt_to_torch(sys.argv[1])