import numpy as np
import matplotlib.pyplot as plt


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


baseName = 'sedov_GP3rd_2quad_RK3_cfl0p8_HLLC_256';dataNumb = '100390'
#baseName = 'sedov_gp5th_3quad_rk4_noDtRed_cfl0p8_hllc_256';dataNumb = '100414'
#baseName = 'sedov_gp7th_4quad_rk4_noDtRed_cfl0p8_hllc_256';dataNumb = '100481'
#baseName = 'sedov_1Dpol1_1quad_FE_noDtRed_cfl0.6_hllc_400_noCSD';dataNumb = '100677'
#baseName = './stabTests/sedov_GPMOOD3_2quad_RK3_cfl0.689_hllc_400';dataNumb = '100644'
#baseName = './stabTests/sedov_GPMOOD3_2quad_RK3_cfl0.59_hllc_400';dataNumb = '100753'
#baseName = './stabTests/sedov_GPMOOD3_2quad_RK3_cfl0.89_hllc_400';dataNumb = '100510'
#baseName = './stabTests/sedov_GPMOOD3_2quad_RK3_cfl0.79_hllc_400';dataNumb = '100570'

#baseName = './stabTests/sedov_GPMOOD3_1quad_RK3_cfl0.85_hllc_400';dataNumb = '100080'

#dataNumb = '101082'
pName    = '_LogDens'
dataName = baseName+'_final_'+dataNumb+'.dat'
plotName = baseName+pName+'_final_'+dataNumb+'.png'
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
plotVar = np.log10(d2.rho)
#plotVar = d2.rho
#plotVar = d2.ordr
#fig1 = plt.figure(figsize=(4,3), dpi=600)
fig1 = plt.figure(figsize=(4,3), dpi=400)
ax = fig1.add_subplot(1, 1, 1)
ax.pcolormesh(xi, yi, plotVar, cmap='jet')   # use 'jet' colormap
fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap='jet') )

# draw a contour lines also
#levels = np.linspace(0.01, 24, 40)   # this is a 1d array with 40 elements, ranging from 0.1 to 1.8.
# will be used for specifying contours
#levels = np.linspace(np.amin(d2.rho), np.amax(d2.rho), 40)
levels = np.linspace(0.004, 4.8, 40)

extent = (np.amin(d2.x), np.amax(d2.x), np.amin(d2.y), np.amax(d2.y))  # this tupple will be used for specifying x, y axis for contours
ax.contour(d2.rho, levels=levels, extent=extent, linewidths=0.2, colors='k')  # 40 contours of 0.2 width, black lines
ax.set_aspect(aspect=1)

#plt.xlim([0.39, 0.61])
#plt.ylim([0.39, 0.61])

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.show()

# save to a file
# to 2d colormap, use bitmap format.
# I prefer 'png'.
fig1.savefig(plotName)
