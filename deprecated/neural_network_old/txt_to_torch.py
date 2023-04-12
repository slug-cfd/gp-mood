import os
from NN import *
import sys
from utils import *

#Takes a txt file from gp mood and output two torch arrays correspond to R=1 an R=0
#for file in ../gp-mood-code/*.txt ; do python3.9 txt_to_torch.py "$file" ;  done

def txt_to_torch(file_name):

    #Print some stuff and compute the paths 
    print(colors.HEADER+"Converting ", file_name, " to torch arrays"+colors.ENDC)

    #This trims i.e.deletes all lines that appear several time in the txt file generated by gp mood
    print("trimming the txt file ...")
    trimmed_path=dir+file_name[len(dir):]
    #os.system('sort '+file_name+'| uniq > "'+trimmed_path+'"')
    print("done")

    #This is where we store the torch arrays
    output_path0 = 'data/'+file_name[len(dir+'trimmed_'):-4]+'_0_torch.pt'
    output_path1 = 'data/'+file_name[len(dir+'trimmed_'):-4]+'_1_torch.pt'

    print(colors.yellow+"will save data to",output_path0,"and",output_path1+colors.ENDC)

    print("enumerating ...")

    N0=0
    N1=0
    #We first count the number of R=0 and R=1 entries to declare the arrays
    with open(trimmed_path, 'r') as file:
        for line in file:

            columns = line.split()
            columns_float = [float(column) for column in columns]
            #If the last digit is 0, we are doing R=0
            if (columns_float[-1]==0):
                N0+=1
            elif (columns_float[-1]==1):
                N1+=1
            else:
                print(colors.red+"ERROR NOT 0 NOR 1")
                sys.exit()



    #Now that the sizes are known, we declare the torch arrays
    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))

    print(N0,"entries with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    print(colors.yellow+"gathering and writing ..."+colors.ENDC)

    #We now distribute the txt data into the torch files
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
                elif (columns_float[-1]==1):
                    for k in range(L):
                        input1[iline1,k]=columns_float[k]
                    iline1+=1
                else:
                    print(colors.red+"ERROR NOT 0 NOR 1")
                    sys.exit()
    print("end of distributing, iline0=",iline0, "N0=",N0)
    print("end of distributing, iline1=",iline1, "N0=",N1)

    #save
    torch.save(input0, output_path0)
    torch.save(input1, output_path1)
    print("end")

if (len(sys.argv)<2):
    print("error, usage: python3 txt_to_torch.py file.txt")
    sys.exit()

txt_to_torch(sys.argv[1])