import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
# making 2d class
class data2d:

    def __init__(self, file_name):  # 'self' means the class itself.
        self.filename = file_name
        self.raw = np.loadtxt(file_name, skiprows=1)  # load txt file using np.loadtxt function
        # for 2d data, we need to 'reshape' each column data
        # to 2d array. so I count every 'unique' entries
        # in x axis for 'xbins', in y axis for 'ybins'.
        self.xbins = int(np.shape(np.unique(self.raw[:, 0]))[0])
        self.ybins = int(np.shape(np.unique(self.raw[:, 1]))[0])
        # now taking each column of data,
        # and reshape to 2d
        # ** and I guess you need to 'transpose' the data
        # since we are handling an array,
        # the first element apperas at 'top-left' corner,
        # which is not the case if we put that in the cartessian coord.
        self.x   = np.reshape(self.raw[:, 0], (self.ybins, self.xbins))
        self.y   = np.reshape(self.raw[:, 1], (self.ybins, self.xbins))
        self.rho = np.reshape(self.raw[:, 2], (self.ybins, self.xbins))
        self.vx  = np.reshape(self.raw[:, 3], (self.ybins, self.xbins))
        self.vy  = np.reshape(self.raw[:, 4], (self.ybins, self.xbins))
        self.p   = np.reshape(self.raw[:, 5], (self.ybins, self.xbins))
        self.ordr= np.reshape(self.raw[:, 6], (self.ybins, self.xbins))


name = sys.argv[1]
var = sys.argv[2]

dataName = name+'.dat'
plotName = name+'_'+var+'.png'

data = data2d(dataName)


def edge_grid(X, Y):

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    xx = X[0, :] - dx/2.
    yy = Y[:, 0] - dy/2.

    xi = np.append(xx, xx[-1]+dx)
    yi = np.append(yy, yy[-1]+dy)

    return xi, yi

xi, yi = edge_grid(data.x, data.y)

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

fig1 = plt.figure(dpi=400)
ax = fig1.add_subplot()
print(np.shape(xi), np.shape(yi))
cmapValue = 'jet'
cmapValue = 'seismic'
ax.set_aspect(1) 
ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue) 
fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue)) 
ax.set_title(var)

fig1.savefig(plotName)
