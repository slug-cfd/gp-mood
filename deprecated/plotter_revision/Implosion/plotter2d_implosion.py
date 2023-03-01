import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
        self.x   = np.reshape(self.raw[:, 0], (self.xbins, self.ybins))
        self.y   = np.reshape(self.raw[:, 1], (self.xbins, self.ybins))
        self.rho = np.reshape(self.raw[:, 2], (self.xbins, self.ybins))
        self.vx  = np.reshape(self.raw[:, 3], (self.xbins, self.ybins))
        self.vy  = np.reshape(self.raw[:, 4], (self.xbins, self.ybins))
        self.p   = np.reshape(self.raw[:, 5], (self.xbins, self.ybins))
        self.ordr= np.reshape(self.raw[:, 6], (self.xbins, self.ybins))


# load data using a class
#d2 = data2d('final.dat')


#baseName = 'implosion_polymood3_2quad_RK3_cfl0.8_HLLC_400'; dataNumb = '108282'
baseName = 'implosion_polymood3_2quad_RK3_cfl0p8_HLLC_400_no'; dataNumb = '107907'
baseName = 'implosion_polymood3_2quad_RK3_cfl0p8_HLLC_400'; dataNumb = '108356'

# 3rd order
#baseName = 'implosion_GP3_2quad_RK3_cfl0.8_HLLC_400_noA_PrioriCheck';         dataNumb = '107902'
#baseName = 'implosion_GP3rd_2quad_RK3_cfl0.8_HLLC_400';         dataNumb = '108326'
#baseName = 'implosion_GP3rd_2quad_RK4_nodtRed_cfl0.8_HLLC_400'; dataNumb = '108357'
#baseName = 'implosion_GP3rd_2quad_RK3_nodtRed_cfl0p8_HLLC_400_divVDx2'; dataNumb = '108330'
baseName = 'implosion_GP3_2quad_RK3_cfl0p8_HLLC_400_no'; dataNumb = '107902'
baseName = 'implosion_GP3rd_2quad_RK3_cfl0p8_HLLC_400'; dataNumb = '108393'

# 5th order
#baseName = 'implosion_GP5th_3quad_RK3_nodtRed_cfl0p8_HLLC_400_divVDx2'; dataNumb = '108535'
#baseName = 'implosion_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_400';   dataNumb ='116445'
baseName = 'implosion_GP5th_3quad_RK3_nodtRed_cfl0p8_HLLC_400'; dataNumb = '108967'

# 7th order
#baseName = 'implosion_GP7th_4quad_RK3_nodtRed_cfl0.8_HLLC_400'; dataNumb ='108504'
#baseName = 'implosion_GP7th_4quad_RK3_nodtRed_cfl0p8_HLLC_400_divVDx2'; dataNumb ='108428'
baseName = 'implosion_GP7th_4quad_RK3_nodtRed_cfl0p8_HLLC_400'; dataNumb = '108715'




dataName = baseName+'_final_'+dataNumb+'.dat'
#plotName = baseName+'_final_'+dataNumb+'.png'
pName    = 'ordr_'
plotName = baseName+'_final_'+pName+dataNumb+'.png'

d2 = data2d(dataName)

cmapChoice='jet'
dpiVal = 400
minVal = 0.3
maxVal = 1.2

def edge_grid(X, Y):

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # shift center points by delta/2
    # this is one dimensional array
    xx = X[0, :] - dx/2.
    yy = Y[:, 0] - dy/2.

    # append last point
    xi = np.append(xx, xx[-1]+dx)
    yi = np.append(yy, yy[-1]+dy)

    return xi, yi

# using the above function, lets take edge grids.

xi, yi = edge_grid(d2.x, d2.y)

# now, redraw a figure
sndSpd=np.sqrt(1.4*d2.p/d2.rho) 
mach = np.sqrt(d2.vx**2 + d2.vy**2)/sndSpd #mach number
plotVar = np.log10(d2.rho)
plotVar = d2.ordr
#plotVar = d2.p
#plotVar = sndSpd
plotVar = mach
#plotVar = d2.rho

fig1 = plt.figure(figsize=(4,3), dpi=dpiVal)
ax = fig1.add_subplot(1, 1, 1)
ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)   # use 'jet' colormap
fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)) ###, vmin=minVal, vmax=maxVal))


# draw a contour lines also
# this is a 1d array with 40 elements, ranging from 0.3 to 1.2 will be used for specifying contours
levels = np.linspace(minVal, maxVal, 40)
extent = (np.amin(d2.x), np.amax(d2.x), np.amin(d2.y), np.amax(d2.y))  # this tupple will be used for specifying x, y axis for contours
ax.contour(plotVar, levels=levels, extent=extent, linewidths=0.2, colors='k') 
ax.set_aspect(aspect=1)

plt.show()

# save to a file
# to 2d colormap, use bitmap format.
# I prefer 'png'.
fig1.savefig(plotName)
