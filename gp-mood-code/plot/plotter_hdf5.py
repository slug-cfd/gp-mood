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
img = axAR.imshow(qqt, interpolation='none',cmap=colmap, extent=[0.5, -0.5, -0.5, 0.5])
axAR.set_title(var)
ticks = np.linspace(qqt.min(), qqt.max(), 5, endpoint=True)
axAR.invert_yaxis()
plt.colorbar(img,ticks=ticks, label=var)

plt.savefig(file[:-3]+'_'+var+'.png',figsize=(4, 3), dpi=100)
fig.clf()
plt.close()
