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

pName = 'ordr_'
dataName = name+'.dat'
plotName = name+'_'+var+'.png'

d2 = data2d(dataName)


def edge_grid(X, Y):

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    xx = X[0, :] - dx/2.
    yy = Y[:, 0] - dy/2.

    xi = np.append(xx, xx[-1]+dx)
    yi = np.append(yy, yy[-1]+dy)

    return xi, yi

xi, yi = edge_grid(d2.x, d2.y)

sndSpd=np.sqrt(1.4*d2.p/d2.rho) 
mach = np.sqrt(d2.vx**2 + d2.vy**2)/sndSpd 

if (var=='ordr'):
    plotVar = d2.ordr
elif(var=='p'):
    plotVar = d2.p
elif(var=='sndSpd'):
    plotVar = sndSpd
elif(var=='mach'):
    plotVar = mach
elif(var=='rho'):
    plotVar = d2.rho
elif(var=='vx'):
    plotVar = d2.vx
elif(var=='vy'):
    plotVar = d2.vy
else:
    print("wrong var")
    sys.exit()

fig1 = plt.figure(figsize=(4,3), dpi=400)
ax = fig1.add_subplot(1, 1, 1)

cmapValue = 'jet'
cmapValue = 'seismic'

print(np.shape(xi))
print(np.shape(yi))
print(np.shape(plotVar))
ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue) 
fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue)) 
print(np.min(d2.rho),np.max(d2.rho))

ax.set_title(var)

fig1.savefig(plotName)
