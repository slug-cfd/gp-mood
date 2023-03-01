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




def edge_grid(X, Y):

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # shift center points by delta/2
    # this is one dimensional array
    xx = X[0, :] - dx/2.
    yy = Y[:, 0] - dy/2.

    #xx = X - 0.5*dx
    #yy = Y - 0.5*dy
    
    # append last point
    xi = np.append(xx, xx[-1]+dx)
    yi = np.append(yy, yy[-1]+dy)

    return xi, yi 
    
# load data using a class
#d2 = data2d('final.dat')


"""
baseName = 'dblmach800-800_GP3_2quad_RK3_cfl0.8_HLL_600x600_noIcHack'
dataNumArr=['10']*3
dataNumArr[0]=dataNumArr[0]+'0566'
dataNumArr[1]=dataNumArr[1]+'1044'
dataNumArr[2]=dataNumArr[2]+'1486'
#dataNumArr[3]=dataNumArr[3]+'2299'
#dataNumArr[4]=dataNumArr[4]+'2870'
#dataNumArr[5]=dataNumArr[5]+'2000'
#dataNumArr[6]=dataNumArr[6]+'2078'
"""

#baseName = 'dmr_gp3_2quad_RK3_nodtRed_cfl0.45_800x200'
#baseName = 'dmr_gp3_2quad_RK3_nodtRed_cfl0p8_800x200'
#baseName = 'dmr_pol-mood3_2quad_RK3_nodtRed_cfl0p8_800x200'
#dataNumArr=['10']*1
#dataNumArr[0]=dataNumArr[0]+'1344'


#baseName ='dmr_gp5_3quad_RK3_nodtRed_cfl0.45_800x200'
baseName = 'dmr_gp5_3quad_RK3_nodtRed_cfl0p8_800x200'
dataNumArr=['10']*1
dataNumArr[0]=dataNumArr[0]+'1355'


#baseName = 'dmr_gp7_4quad_RK3_nodtRed_cfl0p8_800x200_gc4'
#dataNumArr=['10']*1
#dataNumArr[0]=dataNumArr[0]+'1385'


# choose your color scheme (https://matplotlib.org/stable/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py)
cmapChoice = 'jet'
cmapChoice = 'seismic'
#cmapChoice = 'rainbow'
#cmapChoice = 'bwr'
##cmapChoice = 'ocean'
#cmapChoice = 'prism'
#cmapChoice = 'CMRmap'
#cmapChoice = 'Dark2'
#cmapChoice = 'tab20'
#cmapChoice = 'Pastel2'

#set the min and max of the entire data you're plotting
valMin = 1.0
valMax = 23.5

#set the dpi value
dpiVal = 600




ekinArr = np.zeros((6,4))
i=0
ekinArr[:,0] = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006]) #, 0.007])



for index in dataNumArr:

    dataName = baseName+'_final_'+index+'.dat'

    d2 = data2d(dataName)
    xi, yi = edge_grid(d2.x, d2.y)

    # What to plot ==========================
    sndSpd=np.sqrt(1.4*d2.p/d2.rho)
    kinEner = 0.5*d2.rho*(d2.vx**2 + d2.vy**2)
    mach = np.sqrt(d2.vx**2 + d2.vy**2)/sndSpd #mach number
    #plotVar = kinEner
    plotVar = np.log10(d2.rho)
    
    plotVar = d2.ordr
    #plotVar = d2.vy
    #plotVar = sndSpd
    #plotVar = mach
    #plotVar = d2.rho


    pName = 'mach_'
    #dataName = baseName+'_final_'+index+'.dat'
    plotName = baseName+'_final_'+pName+index+'.png'
    

    """
    # making a figure
    fig = plt.figure(figsize=(4,3), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.pcolormesh(d2.x, d2.y, plotVar, shading='nearest',cmap=cmapChoice)
    fig.colorbar(ax.pcolormesh(d2.x, d2.y, plotVar, shading='nearest', cmap=cmapChoice),orientation="horizontal" )
    # set axes ratio properly
    ax.set_aspect(aspect=1)

    # see the result
    plt.show()
    fig.savefig(plotName)
    """
    
    
    # let's plot ==========================

    #fig1 = plt.figure(figsize=(4,3), dpi=600)
    fig1 = plt.figure(figsize=(4,3), dpi=dpiVal)
    ax = fig1.add_subplot(1, 1, 1)
    ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)   # use 'jet' colormap
    #ax.pcolormesh(xi, yi, plotVar, norm=colors.LogNorm(vmin=np.min(d2.rho),vmax=np.max(d2.rho)), cmap=cmapChoice)   # use 'jet' colormap
    #fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)) #, norm=colors.LogNorm(vmin=np.min(d2.rho),vmax=np.max(d2.rho))) )
    #fig1.colorbar(ax.pcolormesh(d2.x-0.5*1./400, d2.y-0.5*1./400, plotVar, cmap=cmapChoice, shading='nearest')) #, norm=colors.LogNorm(vmin=np.min(d2.rho),vmax=np.max(d2.rho))) )
    ##fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice, vmin=np.log10(valMin), vmax=np.log10(valMax)) , orientation="horizontal" )
    fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice), orientation="horizontal" )

    # draw a contour lines ==========================
    levels = np.linspace(0.01, 24, 40)   # this is a 1d array with 40 elements, ranging from 0.1 to 1.8.
    # will be used for specifying contours
    extent = (np.amin(d2.x), np.amax(d2.x), np.amin(d2.y), np.amax(d2.y))  # this tupple will be used for specifying x, y axis for contours
    #extent = (0.01, 160)
    #ax.contour(d2.rho, levels=levels, extent=extent, linewidths=0.2, colors='k')  # 40 contours of 0.2 width, black lines
    ax.set_aspect(aspect=1)
    plt.show()
    #plt.grid()    
    plt.tight_layout()


    # save to a file
    # to 2d colormap, use bitmap format.
    # I prefer 'png'.

    #fig1.savefig(plotName)
    fig1.savefig(plotName,transparent = True, bbox_inches = 'tight', pad_inches = 0)



    # the overall min and max of density = 0.01 and 160
    print('=================================')
    print('min and max of dens')
    print(np.amin(d2.rho),np.amax(d2.rho))
    #print('min and max of ekin')
    #print(np.amin(kinEner),np.amax(kinEner))
    print('')

    """
    # store min and max ekin
    ekinArr[i,2] = np.amin(kinEner)
    ekinArr[i,3] = np.amax(kinEner)

    # write integral quantities
    # cell volume    
    dx = dy = 1.5/600.
    dvol = dx*dy
    ekin = 0.5*sum(sum(d2.rho*(d2.vx**2 + d2.vy**2)))*dvol
    ekinArr[i,1] = ekin
    print('kinetic energy over the domain')
    print(ekinArr[i,0],ekinArr[i,1])
    print('=================================')
    i = i+1
    """

"""
fig = plt.figure(figsize=(4, 3), dpi=dpiVal)
plt.plot(ekinArr[0:i,0],ekinArr[0:i,1],'-r', marker='^',linewidth=0.5, markersize=1.5)
fname = baseName+'_ekin.data'
np.savetxt(fname, ekinArr)
"""

