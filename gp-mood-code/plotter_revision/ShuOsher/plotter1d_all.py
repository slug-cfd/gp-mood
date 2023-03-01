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
        #self.vx  = self.raw[:, 2]     # mx
        #self.vy  = self.raw[:, 3]     # mx
        #self.p   = self.raw[:, 4]      # and pressure.



# now, we can load the data by calling the class that I defined,


dataName1 = 'shuosher1d_GP3rd_1quad_RK3_nodtRed_cfl0.8_HLLC_256_final_100330.dat'

dataName2 = 'shuosher1d_GP5th_1quad_RK4_nodtRed_cfl0.8_HLLC_256_final_100330.dat'
dataName3 = 'shuosher1d_GP5th_1quad_RK4_dtRed_cfl0.8_HLLC_256_final_100542.dat'

dataName4 = 'shuosher1d_GP7th_1quad_RK4_nodtRed_cfl0.8_HLLC_256_final_100331.dat'
dataName5 = 'shuosher1d_GP7th_1quad_RK4_dtRed_cfl0.8_HLLC_256_final_101490.dat'

dataName6 = 'shuosher1d_pol3rd_1quad_RK3_nodtRed_cfl0.8_HLLC_256_final_100330.dat'
#dataName7 = 'shuosher1d_pol3rd_1quad_RK3_nodtRed_cfl0.8_HLLC_4096_final_104955.dat'
dataName7 = 'shuosher1d_GP3rd_1quad_RK3_nodtRed_cfl0.8_HLLC_256_final_100331.dat'

plotName  = 'ShuOsher_3-5-7_noDtRed.png'

        
gp3_rk3_noDt = data1d(dataName1)
gp5_rk4_noDt = data1d(dataName2)
gp5_rk4_Dt   = data1d(dataName3)
gp7_rk4_noDt = data1d(dataName4)
gp7_rk4_Dt   = data1d(dataName5)

pol3  = data1d(dataName6)
ref   = data1d(dataName7) #1024 run


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

#poly-mood
#ax.plot(pol3.x,         pol3.rho,         'm', marker='h',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(pol3.x,         pol3.rho,         'm', marker='x', linewidth=0.5, markersize=3.)

# 3rd-order
#ax.plot(gp3_rk3_noDt.x, gp3_rk3_noDt.rho, 'b', marker='^',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(gp3_rk3_noDt.x, gp3_rk3_noDt.rho, 'b', marker='o', linewidth=0.5, markersize=2.)

# 5th-order
#ax.plot(gp5_rk4_noDt.x, gp5_rk4_noDt.rho, 'g', marker='s',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(gp5_rk4_noDt.x, gp5_rk4_noDt.rho, 'g', marker='2', linewidth=0.5, markersize=4.)
#ax.plot(gp5_rk4_Dt.x,   gp5_rk4_Dt.rho,   'g', marker='s',linewidth=0.5, linestyle='dashed', markersize=1.)

# 7th-order
#ax.plot(gp7_rk4_noDt.x, gp7_rk4_noDt.rho, 'r', marker='h',linewidth=0.5, linestyle='solid', markersize=1.)
ax.plot(gp7_rk4_noDt.x, gp7_rk4_noDt.rho, 'r', marker='<', linewidth=0.5, markersize=2.)
#ax.plot(gp7_rk4_Dt.x,   gp7_rk4_Dt.rho,   'r', marker='d',linewidth=0.5, linestyle='dashed', markersize=1.)

# ref solution with poly-mood on 4096
ax.plot(ref.x,          ref.rho,          'k',            linewidth=0.5, linestyle='solid', markersize=1.)

#ax.plot(dref.x, gp7.rho, color='black', linewidth=1.0)

#ax.set(xlim=(2.7, 3.5), ylim=(0., .8))
ax.set(xlim=(0, 9), ylim=(0., 5.0))
##ax.set(xlim=(4.5, 7.), ylim=(0.5, 5.0))

plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')

#ax.legend([r'  GP-MOOD R=1, RK3, 2 quads', r'GP-MOOD R=2, $\rho$' ,  'Reference'], fontsize = 8)
ax.legend(    [r'POL-MOOD3 w/ RK3', \
               r'GP-MOOD3 (R=1) w/ RK3  ', \
               r'GP-MOOD5 (R=2) w/ RK4 wo $\Delta t$ reduction',\
#               r'GP-MOOD R=2, RK4 w/ dt reduction', \
               r'GP-MOOD7 (R=3) w/ RK4 wo $\Delta t$ reduction', \
#               r'GP-MOOD R=3, RK4 w/ dt reduction', \
               
               r'Reference solution on $N_x=4096$ ', \
               ], fontsize = 8)

#ax.legend([r'  GP-MOOD R=1, RK3, 2 qpts',  r'GP-MOOD R=2, RK3 w/dt red, 3 qpts', r'GP-MOOD R=3, RK4 w/dt red, 4 qpts'], fontsize = 8)




# in order to open a graphical window
# to see the results, you need the following line

# to save a figure, use 'savefig' method under the 'fig'.
# I recommend to use a vector format
# in this case I usued 'pdf'
plt.show()

plt.tight_layout()

fig.savefig(plotName)
