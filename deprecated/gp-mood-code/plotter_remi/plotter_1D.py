import numpy as np
import matplotlib.pyplot as plt
import sys

class data1d:
    def __init__(self, file_name):  # 'self' means the class itself.
        self.filename = file_name
        self.raw = np.loadtxt(file_name)   # load txt file using np.loadtxt function
        self.x = self.raw[:, 0]      # first column would be x-axis
        self.rho = self.raw[:, 1]    # second column would be the density
        self.vx = self.raw[:, 2]     # mx
        self.vy = self.raw[:, 3]     # mx
        self.p = self.raw[:, 4]      # and pressure.


name = sys.argv[1]
var = sys.argv[2]

dataName = name+'.dat'
plotName = name+'_'+var+'.png'
     
data = data1d(dataName)
sndSpd=np.sqrt(1.4*data.p/data.rho) 
mach = np.sqrt(data.vx**2 + data.vy**2)/sndSpd 

if (var=='ordr'):
    plotVar = data.ordr
elif(var=='p'):
    plotVar = data.p
elif(var=='sndSpd'):
    plotVar = sndSpd
elif(var=='mach'):
    plotVar = mach
elif(var=='rho'):
    plotVar = data.rho
elif(var=='vx'):
    plotVar = data.vx
elif(var=='vy'):
    plotVar = data.vy
else:
    print("wrong var")
    sys.exit()

fig = plt.figure(figsize=(4, 3), dpi=600)

ax = fig.add_subplot(1, 1, 1)

ax.plot(data.x, plotVar, '-r', marker='^',linewidth=0.5, markersize=1.5)

#ax.set(xlim=(2.7, 3.5), ylim=(0., .8))

plt.ylabel(var)
plt.xlabel(r'$x$')

plt.tight_layout()

fig.savefig(plotName)
