# lets import essential packages: numpy and matplotlib.
# with nickname of np and plt.
import numpy as np
import matplotlib.pyplot as plt




class data1d:

    # __init__ function will be called once the data1d class called.
    def __init__(self, file_name):  # 'self' means the class itself.
        self.filename = file_name
        self.raw = np.loadtxt(file_name, skiprows=1)   # load txt file using np.loadtxt function
        self.x   = self.raw[:, 0]      # first column would be x-axis
        self.rho = self.raw[:, 2]    # second column would be the density
        self.vx  = self.raw[:, 3]     # mx
        self.vy  = self.raw[:, 4]     # mx
        self.p   = self.raw[:, 5]      # and pressure
        self.ordr   = self.raw[:, 6]      # and pressure.



# now, we can load the data by calling the class that I defined,

"""
dataName1 = 'SS2d_fog_1quad_RK3_cfl0p8_hllc_256x1_final_100212.dat'
dataName2 = 'SS2d_gp3_1quad_RK3_cfl0p8_hllc_256x1_final_100212.dat'
dataName3 = 'SS2d_gp5_1quad_RK3_cfl0p8_hllc_256x1_final_100212.dat'
dataName4 = 'SS2d_gp7_1quad_RK3_cfl0p8_hllc_256x1_final_100212.dat'
"""


dataName1 = 'SS2d_fog_1quad_RK3_cfl0p8_hllc_256x1_icNew_final_100134.dat'
dataName2 = 'SS2d_gp3_1quad_RK3_cfl0p8_hllc_256x1_icNew_final_100137.dat'
dataName3 = 'SS2d_gp5_1quad_RK3_cfl0p8_hllc_256x1_icNew_final_100137.dat'
dataName4 = 'SS2d_gp7_1quad_RK3_cfl0p8_hllc_256x1_icNew_final_100137.dat'

plotName  = 'slow_moving_shock.png'

        
data1 = data1d(dataName1)
data2 = data1d(dataName2)
data3 = data1d(dataName3)
data4 = data1d(dataName4)

# then,
# d1.x == x-axis for the file
# d1.rho == density data
# ...


# now we have data, lets draw a figure.
# by using matplotlib.

# initialize a figure with
# 4x3 (inches) and
# 600 dpi. higher dpi makes higher resolution figure.
# figsize=(4,3), dpi=600 is my usual setting.
fig = plt.figure(figsize=(4, 3), dpi=400)

# add a subplot to the figure 'fig'.
# 1, 1, 1 means, in 1x1 grid, the first subplot.
ax = fig.add_subplot(1, 1, 1)

# plot function in matplotlib draw a line with given data.
# below example, d1.x becomes a horizontal axis,
# and d1.rho becomes the vertical axis.

ax.plot(data1.x, data1.rho, 'r', marker='x', linewidth=0.5, markersize=1.)
#ax.plot(data1.x, data1.ordr, 'r', marker='x', linewidth=0.5, markersize=1.)

ax.plot(data2.x, data2.rho, 'g', marker='o', linewidth=0.5, markersize=1.)
#ax.plot(data2.x, data2.ordr, 'g', marker='o', linewidth=0.5, markersize=1.)

ax.plot(data3.x, data3.rho, 'b', marker='+', linewidth=0.5, markersize=1.)
#ax.plot(data3.x, data3.ordr, 'b', marker='+', linewidth=0.5, markersize=1.)

ax.plot(data4.x, data4.rho, 'c', marker='d', linewidth=0.5, markersize=1.)
ax.plot(data4.x, data4.ordr, 'c', marker='d', linewidth=0.5, markersize=1.)

#ax.set(xlim=(4.5, 7.), ylim=(0.5, 5.0))

plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')



# calculate an exact solution
# shock velocity
dL=5.6698
uL=-1.4701
pL=100.
dR=1.0
uR=-2.315811
pR=1.0

# shock velocity
vs = (dR*uR-dL*uL)/(dR-dL)

# eigenval at Left
aL = np.sqrt(1.4*pL/dL)
lambL1 = uL - aL
lambL2 = uL
lambL3 = uL + aL

# eigenval at Right
aR = np.sqrt(1.4*pR/dR)
lambR1 = uR - aR
lambR2 = uR
lambR3 = uR + aR

# shock travel time 
dt = 1.

# shock location
shockLoc = 15.+ vs*dt

data_exact = data1.rho
for i in range(len(data1.x)):
    if data1.x[i] < shockLoc:
        data_exact[i] = 5.6698
    else:
        data_exact[i] = 1.

ax.plot(data1.x, data_exact, 'k', linewidth=1.)
        
#ax.legend([r'  GP-MOOD R=1, RK3, 2 quads', r'GP-MOOD R=2, $\rho$' ,  'Reference'], fontsize = 8)
ax.legend(    [r'data1',\
               r'data2',\
               r'data3',\
               r'data4',\
               r'exact',\
               ], fontsize = 8)



# in order to open a graphical window
# to see the results, you need the following line

# to save a figure, use 'savefig' method under the 'fig'.
# I recommend to use a vector format
# in this case I usued 'pdf'
plt.show()

plt.tight_layout()

fig.savefig(plotName)
