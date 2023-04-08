import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

file = sys.argv[1]
var  = sys.argv[2]

f = h5py.File(file)

colmap='inferno'

qqt   = np.array(f[var])

ny=qqt.shape[0]
nx=qqt.shape[1]

print(qqt)

fig, (axAR) = plt.subplots(1, 1)
img = axAR.imshow(qqt, interpolation='none',cmap=colmap, extent=[0.5, -0.5, -0.5, 0.5])
axAR.set_title(var)
ticks = np.linspace(qqt.min(), qqt.max(), 5, endpoint=True)
#axAR.invert_xaxis()
plt.colorbar(img,ticks=ticks, label=var)

plt.savefig(file[:-3]+'_'+var+'.png',figsize=(16, 12), dpi=200)
fig.clf()
plt.close()
