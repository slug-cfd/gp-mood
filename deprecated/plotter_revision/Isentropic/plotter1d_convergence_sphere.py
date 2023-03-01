# lets import essential packages: numpy and matplotlib.
# with nickname of np and plt.
import numpy as np
import matplotlib.pyplot as plt




class data1d:

    # __init__ function will be called once the data1d class called.
    def __init__(self, file_name):  # 'self' means the class itself.
        self.filename = file_name
        self.raw = np.loadtxt(file_name)   # load txt file using np.loadtxt function
        self.x   = self.raw[-1,0]      # first column would be x-axis
        self.y   = self.raw[-1,1]      # second column would be the density
        self.L1  = self.raw[-1,2]     # mx
        self.inv = self.raw[-1,3]     # mx



# now, we can load the data by calling the class that I defined,
dataName1 = 'isen_GP3rd_2quad_RK3_cfl0.8_HLLC_Lx20_050_ell_1.0_error.dat'
dataName2 = 'isen_GP3rd_2quad_RK3_cfl0.8_HLLC_Lx20_100_ell_1.0_error.dat'
dataName3 = 'isen_GP3rd_2quad_RK3_cfl0.8_HLLC_Lx20_200_ell_1.0_error.dat'
dataName4 = 'isen_GP3rd_2quad_RK3_cfl0.8_HLLC_Lx20_400_ell_1.0_error.dat'



dataName11 = 'isen_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_Lx20_050_ell_1.0_error.dat'
dataName12 = 'isen_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_Lx20_100_ell_1.0_error.dat'
dataName13 = 'isen_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_Lx20_200_ell_1.0_error.dat'
dataName14 = 'isen_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_Lx20_400_ell_1.0_error.dat'
#dataName15 = 'isen_GP5th_3quad_RK4_dtRed_cfl0.8_HLLC_512_ell_1.0_error.dat'


dataName21 = 'isen_GP7th_4quad_RK4_dtRed_cfl0.8_HLLC_Lx20_050_ell_1.0_error.dat'
dataName22 = 'isen_GP7th_4quad_RK4_dtRed_cfl0.8_HLLC_Lx20_100_ell_1.0_error.dat'
dataName23 = 'isen_GP7th_4quad_RK4_dtRed_cfl0.8_HLLC_Lx20_200_ell_1.0_error.dat'
dataName24 = 'isen_GP7th_4quad_RK4_dtRed_cfl0.8_HLLC_Lx20_400_ell_1.0_error.dat'
#dataName25 = 'isen_GP7th_4quad_RK4_dtRed_cfl0.8_HLLC_Lx20_400_ell_1.0_error.dat'


plotName = 'isentropic_conv_sphere.png'


# 3rd order + 2 quad + RK3
gp1 = data1d(dataName1)
gp2 = data1d(dataName2)
gp3 = data1d(dataName3)
gp4 = data1d(dataName4)
#gp5 = data1d(dataName5)

grids    = np.array([gp1.x, gp2.x, gp3.x, gp4.x]) #, gp5.x]
gp3_2_L1_org = np.array([gp1.L1, gp2.L1, gp3.L1, gp4.L1]) #, gp5.L1]


# 5th order + 3 quad + RK4 w/ dt red (correct order)
gp11 = data1d(dataName11)
gp12 = data1d(dataName12)
gp13 = data1d(dataName13)
gp14 = data1d(dataName14)
#gp15 = data1d(dataName15)

gp5_3_L1_org = np.array([gp11.L1, gp12.L1, gp13.L1, gp14.L1]) #, gp20.L1]


# 7th order + 4 quad + RK4 w/ dt red (correct order)
gp21 = data1d(dataName21)
gp22 = data1d(dataName22)
gp23 = data1d(dataName23)
gp24 = data1d(dataName24)
#gp25 = data1d(dataName25)

gp7_4_L1_org = np.array([gp21.L1, gp22.L1, gp23.L1, gp24.L1]) #, gp29.L1] #, gp30.L1]



# compute EOC before doing anything else
#gp3_2_L1



# convert data to log10 scale
gridsLog    = np.log10(grids)
gp3_2_L1 = np.log10(gp3_2_L1_org)
gp5_3_L1 = np.log10(gp5_3_L1_org)
gp7_4_L1 = np.log10(gp7_4_L1_org)

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

# add a subplot to the figure 'fig'.
# 1, 1, 1 means, in 1x1 grid, the first subplot.
#ax1 = fig1.add_subplot(1, 1, 1)

# plot function in matplotlib draw a line with given data.
# below example, d1.x becomes a horizontal axis,
# and d1.rho becomes the vertical axis.
##ax1.plot(grids, gp3_2_L1, '-r', marker='^',linewidth=0.5, markersize=1.5)
##ax1.plot(grids, gp5_3_L1, '-g', marker='s',linewidth=0.5, markersize=1.5)
##ax1.plot(grids, gp7_4_L1, '-b', marker='o',linewidth=0.5, markersize=1.5)

fig2 = plt.figure(figsize=(4, 3), dpi=400)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(gridsLog, gp3_2_L1, 'b', marker='^',linestyle = 'None', markersize=4)
ax2.plot(gridsLog, gp5_3_L1, 'g', marker='s',linestyle = 'None', markersize=4)
ax2.plot(gridsLog, gp7_4_L1, 'r', marker='>',linestyle = 'None', markersize=4)

ax2.plot(gridsLog, -np.log10(grids**3)+5.3, '--b', linewidth=0.5)
ax2.plot(gridsLog, -np.log10(grids**5)+8.2, '--g', linewidth=0.5)
ax2.plot(gridsLog, -np.log10(grids**7)+11.6, '--r', linewidth=0.5)

#ax.set(xlim=(2.7, 3.5), ylim=(0., .8))


plt.xlabel(r'Grid Resolution')
plt.xticks(gridsLog,[r'$50$',r'$100$',r'$200$',r'$400$'])

plt.ylabel(r'$L_1$ error')
#plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1],[r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
plt.yticks([-7, -5, -3, -1, 1],[r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$', r'$10^{1}$'])

#plt.legend([r'3rd-order GP-MOOD3, $R=1$', r'5th-order GP-MOOD5, $R=2$', r'7th-order GP-MOOD7, $R=3$'], fontsize = 8)
plt.legend([r'3rd-order GP-MOOD3', r'5th-order GP-MOOD5', r'7th-order GP-MOOD7'], fontsize = 8)
plt.tight_layout()
#plt.grid()
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)



# in order to open a graphical window
# to see the results, you need the following line

# to save a figure, use 'savefig' method under the 'fig'.
# I recommend to use a vector format
# in this case I usued 'pdf'
plt.show()
plt.tight_layout()

#fig1.savefig(plotName)
fig2.savefig(plotName)
