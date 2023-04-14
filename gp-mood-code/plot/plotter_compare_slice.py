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

file_list=sys.argv[1:-1]
var  = sys.argv[-1]
print(file_list, var)

fig, (axAR) = plt.subplots(1, 1)


for file in file_list:

    f = h5py.File(file)
    qqt   = np.array(f[var])
    method=sum_char(f["method"])
    problem=sum_char(f["problem"][:])

    nx= qqt.shape[0]
    ny= qqt.shape[1]

    slice_qqt=np.zeros(nx)

    for i in range(nx):
        slice_qqt[i]=qqt[128,i]

    plt.plot(slice_qqt, label=method)

plt.xlabel('slice axis')
plt.ylabel(var)
axAR.set_title('slice '+problem+' '+var)
plt.legend(fontsize=3)
plt.savefig('compare_slice_'+problem+'_'+var+'.png',figsize=(8, 6), dpi=200)
fig.clf()
plt.close()

