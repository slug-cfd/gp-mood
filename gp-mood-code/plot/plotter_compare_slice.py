import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

def sum_char(input):
    r=''
    for char in input:
        r=r+str(char)[-2]
    return r

def format(x):
    """Format a double with one digit after the comma"""
    return f"{x:.5f}"

file_list=sys.argv[1:-2]
var  = sys.argv[-2]
title= sys.argv[-1]
print(file_list, var, title)

fig, (axAR) = plt.subplots(1, 1)


for file in file_list:

    f = h5py.File(file)
    qqt   = np.array(f[var])
    method=sum_char(f["method"])
    problem=sum_char(f["problem"][:])

    nx= qqt.shape[0]
    ny= qqt.shape[1]

    if (problem=="shu_osher"):
        L=10*np.sqrt(2)*2
        Ldiag=L*np.sqrt(2)
        x=np.linspace(0,L,nx)
        beg=int(0) 
        end=int(nx/4)
        xlabel="x=y"

    num_NN=""
    if (method[0:13]=="NN_GP_MOOD_CC"):

        num_NN=method[-3]
        method="NN_GP_MOOD_CC, NN #"+num_NN



    slice_qqt=np.zeros(nx)

    for i in range(nx):
        slice_qqt[i]=qqt[i,i]

    plt.plot(x[beg:end], slice_qqt[beg:end], label=method)

plt.xlabel(xlabel)
plt.ylabel(var)
axAR.set_title('slice '+problem+' '+var)
plt.legend(fontsize=9)
plt.savefig(title+num_NN+'_'+var+'.png',figsize=(8, 6), dpi=200)
fig.clf()
plt.close()


#Shu osher
# for file in output_* ; do python3.9 ../../plotter_compare_slice.py   output_shu_osher_GP_MOOD_CFL_0.8_512_512_100324.h5 output_shu_osher_FOG_CFL_0.8_512_512_100322.h5 $file rho compare_FOG_GPMOOD_NN_GP_MOOD; done