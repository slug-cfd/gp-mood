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
    
# load data using a class
#d2 = data2d('final.dat')

#baseName = 'Mach1600_GP3rd_1quad_RK3_cfl0.8_HLLC_600';dataNumb = '101243'
#baseName = 'Mach1600_GP3rd_2quad_RK3_cfl0.8_HLLC_600';dataNumb = '101054'
#baseName = 'Mach1600_GP5th_2quad_RK3_cfl0.8_HLLC_600';dataNumb = '101025'
#baseName = 'Mach1600_GP5th_3quad_RK3_cfl0.8_HLLC_600';dataNumb = '101023'
#baseName = 'Mach1600_GP7th_4quad_RK3_cfl0.8_HLLC_600';dataNumb = '101016'
#baseName = 'Mach1600_GP7th_4quad_RK4_cfl0.8_HLLC_600';dataNumb = '101035'

#baseName = 'DblMach1600_GP7th_4quad_RK4_cfl0.8_HLLC_600';dataNumb = '101109'
#baseName = 'DblMach1600_GP7th_3quad_RK4_cfl0.8_HLLC_600';dataNumb = '101092'
#baseName = 'DblMach1600_GP7th_3quad_RK3_cfl0.8_HLLC_600';dataNumb = '101094'
#baseName = 'DblMach1600_GP7th_4quad_RK3_cfl0.8_HLLC_600';dataNumb = '101080'

#baseName = 'DblMach1600_GP3rd_1quad_RK3_cfl0.8_HLLC_600';dataNumb = '101145'
#baseName = 'DblMach1600_GP3rd_2quad_RK3_cfl0.8_HLLC_600';dataNumb = '101080'


#baseName = 'Mach100fixedIC_GP3th_1quad_RK3_cfl0.8_HLL_600';dataNumb = '102217'
#baseName = 'Mach100fixedIC_GP3th_2quad_RK3_cfl0.8_HLL_600';dataNumb = '102231'
#baseName = 'Mach100fixedIC_GP5th_2quad_RK3_cfl0.8_HLLC_600';dataNumb = '102755'
#baseName = 'Mach100fixedIC_GP5th_3quad_RK3_cfl0.8_HLL_600';dataNumb = '102263'
#baseName = 'Mach100fixedIC_GP7th_3quad_RK3_cfl0.8_HLL_600'; dataNumb = '102286'

# double mach jets 100-800 collision
baseName = 'DblMach100-800_GP5th_3quad_RK3_cfl0.8_HLL_600';dataNumb = '102697'; dataNumb = '101304'
baseName = 'DblMach100-800_GP7th_4quad_RK3_cfl0.8_HLL_600';dataNumb = '102803'; dataNumb = '101315'

# double mach jets 800-800 collision
# times are t=0.002; t=0.003; t=0.004; t=0.005
#baseName = 'DblMach800-800_GP7th_2quad_RK3_cfl0.8_HLL_600'; dataNumb = '102668'; #dataNumb = '102071'; #dataNumb = '101529'; #dataNumb = '100994'
baseName = 'DblMach800-800_GP5th_2quad_RK3_cfl0.8_HLL_600'; dataNumb = '102573'; #dataNumb = '101965'; #dataNumb = '101435'; #dataNumb = '100980'
#baseName = 'DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600'; dataNumb = '102658'; dataNumb = '101980'; dataNumb = '101486'; dataNumb = '101044'

#compute kin energy for DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600

# [1] for DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600 data
# DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600.log
baseName = 'DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600';
# t=0.001 --> 100566
# t=0.002 --> 101044
# t=0.003 --> 101486
# t=0.004 --> 101980

dataNumArr=['10']*5
dataNumArr[0]=dataNumArr[0]+'0566'
dataNumArr[1]=dataNumArr[1]+'1044'
dataNumArr[2]=dataNumArr[2]+'1486'
dataNumArr[3]=dataNumArr[3]+'1980'
dataNumArr[4]=dataNumArr[4]+'2658'
#dataNumArr[5]=dataNumArr[5]+'2668'
#dataNumArr[6]=dataNumArr[6]+'3256'

"""
# [2] for dblmach800-800_GP5_3quad_RK3_cfl0.8_HLL_600_new data
# dblmach800-800_GP5_3quad_RK3_cfl0.8_HLL_600x600_new.log
baseName = 'dblmach800-800_GP5_3quad_RK3_cfl0.8_HLL_600_new'
# t=0.001 --> 100496
# t=0.002 --> 101127
# t=0.003 --> 101608
# t=0.004 --> 102105
# t=0.005 --> 102666

dataNumArr=['10']*5
dataNumArr[0]=dataNumArr[0]+'0496'
dataNumArr[1]=dataNumArr[1]+'1127'
dataNumArr[2]=dataNumArr[2]+'1608'
dataNumArr[3]=dataNumArr[3]+'2105'
dataNumArr[4]=dataNumArr[4]+'2666'
#dataNumArr[5]=dataNumArr[5]+'2668'
#dataNumArr[6]=dataNumArr[6]+'3256'
"""


# [3] for dblmach800-800_GP7_4quad_RK3_cfl0.8_HLL_600_new data
# dblmach800-800_GP7_4quad_RK3_cfl0.8_HLL_600x600_new.log
baseName = 'dblmach800-800_GP7_4quad_RK3_cfl0.8_HLL_600_new'
# t=0.001 --> 100473
# t=0.002 --> 100930
# t=0.003 --> 101422
# t=0.004 --> 101937
# t=0.005 --> 102486

dataNumArr=['10']*5
dataNumArr[0]=dataNumArr[0]+'0473'
dataNumArr[1]=dataNumArr[1]+'0930'
dataNumArr[2]=dataNumArr[2]+'1422'
dataNumArr[3]=dataNumArr[3]+'1937'
dataNumArr[4]=dataNumArr[4]+'2486'
#dataNumArr[5]=dataNumArr[5]+'2668'
#dataNumArr[6]=dataNumArr[6]+'3256'




"""
dataNumArr=['10']*7
dataNumArr[0]=dataNumArr[0]+'0000'
dataNumArr[1]=dataNumArr[1]+'0488'
dataNumArr[2]=dataNumArr[2]+'0994'
dataNumArr[3]=dataNumArr[3]+'1529'
dataNumArr[4]=dataNumArr[4]+'2071'
dataNumArr[5]=dataNumArr[5]+'2668'
dataNumArr[6]=dataNumArr[6]+'3256'
"""

# regular mach 100 light jet
#baseName = 'Mach100_GP3rd_2quad_RK3_cfl0.8_HLL_600'; dataNumb = '102810'
#baseName = 'Mach100_GP5th_3quad_RK3_cfl0.8_HLL_600'; dataNumb = '102687'
#baseName = 'Mach100_GP7th_4quad_RK3_cfl0.8_HLL_600'; dataNumb = '102823'


# choose your color scheme (https://matplotlib.org/stable/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py)
cmapChoice = 'jet'
#cmapChoice = 'ocean'
#cmapChoice = 'Dark2'

#set the min and max of the entire data you're plotting
valMin = 0.004
valMax = 190.

#set the dpi value
dpiVal = 400

for index in dataNumArr:

    dataName = baseName+'_final_'+index+'.dat'
    #plotName = baseName+'_final_'+dataNumb+'.png'
    plotName = baseName+'_final_noCntr'+index+'.png'


    # using the above function, lets take edge grids.
    d2 = data2d(dataName)
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
    #fig1 = plt.figure(figsize=(4,3), dpi=600)
    fig1 = plt.figure(figsize=(4,3), dpi=dpiVal)
    ax = fig1.add_subplot(1, 1, 1)
    ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)   # use 'jet' colormap
    #ax.pcolormesh(xi, yi, plotVar, norm=colors.LogNorm(vmin=np.min(d2.rho),vmax=np.max(d2.rho)), cmap='jet')   # use 'jet' colormap
    #fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice, vmin=np.log10(valMin), vmax=np.log10(valMax)) )
    fig1.colorbar(ax.pcolormesh(xi, yi, plotVar, cmap=cmapChoice)) #, vmin=np.log10(valMin), vmax=np.log10(valMax)) )

    # draw a contour lines also
    levels = np.linspace(0.01, 24, 40)   # this is a 1d array with 40 elements, ranging from 0.1 to 1.8.
    # will be used for specifying contours
    extent = (np.amin(d2.x), np.amax(d2.x), np.amin(d2.y), np.amax(d2.y))  # this tupple will be used for specifying x, y axis for contours
    #extent = (0.01, 160)
    #ax.contour(d2.rho, levels=levels, extent=extent, linewidths=0.2, colors='k')  # 40 contours of 0.2 width, black lines
    ax.set_aspect(aspect=1)

    print('min and max of density')
    print(np.amin(d2.rho),np.amax(d2.rho))
    # the overall min and max of density = 0.01 and 160

    plt.show()

    # save to a file
    # to 2d colormap, use bitmap format.
    # I prefer 'png'.

    fig1.savefig(plotName)

