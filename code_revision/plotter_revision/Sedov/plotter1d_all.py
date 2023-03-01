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
        self.rho = self.raw[:, 1]    # second column would be the density
        #self.vx  = self.raw[:, 2]     # mx
        #self.vy  = self.raw[:, 3]     # mx
        #self.p   = self.raw[:, 4]      # and pressure.



# now, we can load the data by calling the class that I defined,


dataName1 = 'sedov_GP3rd_2quad_RK3_cfl0p8_HLLC_256_slice_y.dat'
dataName2 = 'sedov_GP3rd_2quad_RK3_cfl0p8_HLLC_256_slice_xy.dat'

dataName3 = 'sedov_gp5th_3quad_rk4_noDtRed_cfl0p8_hllc_256_slice_y.dat'
dataName4 = 'sedov_gp5th_3quad_rk4_noDtRed_cfl0p8_hllc_256_slice_xy.dat'


dataName5 = 'sedov_gp7th_4quad_rk4_noDtRed_cfl0p8_hllc_256_slice_y.dat'
dataName6 = 'sedov_gp7th_4quad_rk4_noDtRed_cfl0p8_hllc_256_slice_xy.dat'


sedov_3rd_y  = data1d(dataName1)
sedov_3rd_xy = data1d(dataName2)

sedov_5th_y  = data1d(dataName3)
sedov_5th_xy = data1d(dataName4)

sedov_7th_y  = data1d(dataName5)
sedov_7th_xy = data1d(dataName6)

plotName  = 'sedov_5th-order_cuts.png'

# initialize a figure with
# 4x3 (inches) and
# 600 dpi. higher dpi makes higher resolution figure.
# figsize=(4,3), dpi=600 is my usual setting.

#shift factor
dd=np.sqrt(0.5)-0.5

'''
# 3rd-order
fig = plt.figure(figsize=(4, 3), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.plot(sedov_3rd_y.x,     sedov_3rd_y.rho,  'r', marker='^',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(sedov_3rd_xy.x-dd, sedov_3rd_xy.rho, 'b', marker='s',linewidth=0.5, linestyle='solid', markersize=1.)
ax.set(xlim=(0, 1), ylim=(-0.2, 6.))
plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
ax.legend([r'3rd-order GP-MOOD $y=0$', r'3rd-order GP-MOOD $y=x$'], fontsize = 8)
'''

#'''
#5th-order
fig = plt.figure(figsize=(4, 3), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.plot(sedov_5th_y.x,     sedov_5th_y.rho,  'r', marker='^',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(sedov_5th_xy.x-dd, sedov_5th_xy.rho, 'b', marker='s',linewidth=0.5, linestyle='solid', markersize=1.)
ax.set(xlim=(0, 1), ylim=(-0.2, 6.0))
plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
ax.legend([r'5th-order GP-MOOD $y=0$', r'5th-order GP-MOOD $y=x$'], fontsize = 8)
#'''

'''
# 7th-order
fig = plt.figure(figsize=(4, 3), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.plot(sedov_7th_y.x,     sedov_7th_y.rho,  'r', marker='^',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(sedov_7th_xy.x-dd, sedov_7th_xy.rho, 'b', marker='s',linewidth=0.5, linestyle='solid', markersize=1.)
ax.set(xlim=(0, 1), ylim=(-0.2, 6.))
plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
ax.legend([r'7th-order GP-MOOD $y=0$', r'7th-order GP-MOOD $y=x$'], fontsize = 8)
'''




# in order to open a graphical window
# to see the results, you need the following line

# to save a figure, use 'savefig' method under the 'fig'.
# I recommend to use a vector format
# in this case I usued 'pdf'
plt.show()

plt.tight_layout()

fig.savefig(plotName)
