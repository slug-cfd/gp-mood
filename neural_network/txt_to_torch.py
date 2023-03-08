import os
from NN import *
import sys
L=57

def txt_to_torch(intput_path):
    output_path0 = path[0:-4]+'_0_torch.pt'
    output_path1 = path[0:-4]+'_1_torch.pt'

    print("saving ", intput_path, "to", output_path0)

    N0=0
    N1=0

    with open(path, 'r') as file:
        for line in file:

            columns = line.split()
            columns_float = [float(column) for column in columns]

            if (columns_float[-1]==0):
                N0+=1
            else:
                N1+=1


    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))

    print(N0,"R=0", N1, "R=1", intput_path)

    with open(path, 'r') as file:
            
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

# List the datafiles
# path = '../gp-mood-code/'
# paths = ['../gp-mood-code/trimmed_TD_DMACH___CFL_0.8.txt']
# # for file_name in os.listdir(path):
# #     if "trimmed" in file_name:
# #         paths.append(path+file_name)

#for path in paths:
#     txt_to_torch(path)

txt_to_torch(sys.argv[1])