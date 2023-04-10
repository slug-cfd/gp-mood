import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

file = sys.argv[1]
var  = sys.argv[2]

f = h5py.File(file)

colmap='inferno'

qqt   = np.array(f[var])

fig, (axAR) = plt.subplots(1, 1)

nx= qqt.shape[0]
ny= qqt.shape[1]

slice_qqt=np.zeros(nx)

for i in range(nx):
    slice_qqt[i]=qqt[i,i]


plt.plot(slice_qqt, label=var)
plt.xlabel('slice axis')
plt.ylabel(var)
axAR.set_title('slice_'+file[:-3]+'_'+var)
plt.legend()

plt.savefig(file[:-3]+'_slice_'+var+'.png',figsize=(4, 3), dpi=100)
fig.clf()
plt.close()
