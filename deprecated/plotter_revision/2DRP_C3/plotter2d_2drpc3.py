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
        self.x   = np.reshape(self.raw[:, 0], (self.ybins, self.xbins))
        self.y   = np.reshape(self.raw[:, 1], (self.ybins, self.xbins))
        self.rho = np.reshape(self.raw[:, 2], (self.ybins, self.xbins))
        self.vx  = np.reshape(self.raw[:, 3], (self.ybins, self.xbins))
        self.vy  = np.reshape(self.raw[:, 4], (self.ybins, self.xbins))
        self.p   = np.reshape(self.raw[:, 5], (self.ybins, self.xbins))
        self.ordr= np.reshape(self.raw[:, 6], (self.ybins, self.xbins))


# load data using a class
#d2 = data2d('final.dat')

#baseName = '2drp_pol-mood3_2quad_RK3_noDtRed_cfl0p8_HLLC_400';dataNumb = '100996'
#baseName = '2drp_GP3rd_2quad_RK3_noDtRed_cfl0p8_HLLC_400'; dataNumb = '101000'
baseName = '2drp_GP5th_3quad_RK4_noDtRed_cfl0p8_HLLC_400'; dataNumb = '101037'
#baseName = '2drp_GP7th_4quad_RK4_noDtRed_cfl0p8_HLLC_400'; dataNumb = '101038'
#baseName = '2drp_GP3rd_2quad_RK3_noDtRed_cfl0p8_HLLC_400_noCSD'; dataNumb = '100947'
#baseName = '2drp_pol-mood3_2quad_RK3_noDtRed_cfl0p8_HLLC_400_noCSD'; dataNumb = '100942'
#baseName = '2drp_GP5th_3quad_RK4_noDtRed_cfl0p8_HLLC_400_noCSD'; dataNumb = '100964'

pName = 'ordr_'
dataName = baseName+'_final_'+dataNumb+'.dat'
plotName = baseName+'_final_'+pName+dataNumb+'.png'
#plotName = baseName+'_final_noCntr'+dataNumb+'.png'

d2 = data2d(dataName)
#d2 = np.transpose(d2)
#dataFile = "../output/results/2DRP_conf3/o5/divv_3quad_rk4_final.dat"
#d2 = data2d(dataFile)

# making a figure
#fig = plt.figure(figsize=(4,3), dpi=600)
# subplot
#ax = fig.add_subplot(1, 1, 1)
# now, lets draw a colormap.
# I use 'pcolormesh' function
# inputs are x, y axes and the data; rho
#ax.pcolormesh(d2.x, d2.y, d2.rho)
# set axes ratio properly
#ax.set_aspect(aspect=1)

# see the result
#plt.show()





# however, this example is little bit wrong.
# since 'pcolormesh' function takes
# 'edge' of each pixel, not the 'center'.
# from the original documents,
# ax.pcolormesh(X, Y, C) ==
# (X[i+1, j], Y[i+1, j])          (X[i+1, j+1], Y[i+1, j+1])
#                       +--------+
#                       | C[i,j] |
#                       +--------+
#     (X[i, j], Y[i, j])          (X[i, j+1], Y[i, j+1]),
#
# therefore, we need to make a function
# to calculate 'edges'.

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
#plotVar = mach
#plotVar = d2.rho
#plotVar = d2.vx
#plotVar = d2.vy
#plotVar = d2.p
minval = 0.1
maxval = 1.8
#fig1 = plt.figure(figsize=(4,3), dpi=600)
fig1 = plt.figure(figsize=(4,3), dpi=400)
ax = fig1.add_subplot(1, 1, 1)

cmapValue = 'jet'
cmapValue = 'seismic'

ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue) ##, vmin=minval,vmax=maxval)   # use 'jet' colormap
##ax.pcolormesh(xi, yi, plotVar, norm=colors.LogNorm(vmin=minval,vmax=maxval), cmap='seismic')   # use 'jet' colormap
#ax.pcolormesh(xi, yi, plotVar, norm=colors.LogNorm(vmin=np.min(d2.rho),vmax=np.max(d2.rho)), cmap='jet')   # use 'jet' colormap
fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapValue)) ###,vmin=minval,vmax=maxval) )
#fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap='jet',norm=colors.LogNorm(vmin=minval,vmax=maxval)) )
print(np.min(d2.rho),np.max(d2.rho))

# draw a contour lines also
levels = np.linspace(minval,maxval, 40) 
extent = (np.amin(d2.x), np.amax(d2.x), np.amin(d2.y), np.amax(d2.y))  # this tupple will be used for specifying x, y axis for contours
ax.contour(d2.rho, levels=levels, extent=extent, linewidths=0.2, colors='k')  # 40 contours of 0.2 width, black lines
ax.set_aspect(aspect=1)

plt.show()

# save to a file
# to 2d colormap, use bitmap format.
# I prefer 'png'.

fig1.savefig(plotName)


"""
pcm = ax[0].pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[1], extend='max')
"""
