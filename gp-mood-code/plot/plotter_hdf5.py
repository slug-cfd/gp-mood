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

ratio = nx/ny
img = axAR.imshow(qqt, interpolation='none',cmap=colmap, extent=[0.5, -0.5, -0.5*ratio, 0.5*ratio])
axAR.set_title(file[:-3]+'_'+var)
ticks = np.linspace(qqt.min(), qqt.max(), 5, endpoint=True)
axAR.invert_yaxis()
plt.colorbar(img,ticks=ticks, label=var)

plt.savefig(file[:-3]+'_'+var+'.png',figsize=(4, 3), dpi=100)
fig.clf()
plt.close()
