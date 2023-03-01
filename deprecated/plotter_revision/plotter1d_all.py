# lets import essential packages: numpy and matplotlib.
# with nickname of np and plt.
import numpy as np
import matplotlib.pyplot as plt




class data1d:

    # __init__ function will be called once the data1d class called.
    def __init__(self, file_name):  # 'self' means the class itself.
        self.filename = file_name
        self.raw = np.loadtxt(file_name)   # load txt file using np.loadtxt function
        self.x = self.raw[:, 0]      # first column would be x-axis
        self.rho = self.raw[:, 1]    # second column would be the density
        self.mx = self.raw[:, 2]     # mx
        self.my = self.raw[:, 3]     # mx
        self.p = self.raw[:, 4]      # and pressure.



# now, we can load the data by calling the class that I defined,

dataName1 = 'ShuOsher_2quad_rk3_3rdDivvDMP_400_slice_xy.dat '
dataName2 = 'ShuOsher_3quad_rk3_5thDivvDMP_400_slice_xy.dat'
dataName3 = 'ShuOsher_4quad_rk3_7thDivvDMP_400_slice_xy.dat'
plotName = 'ShuOsher_3-5-7.png'

        
gp3 = data1d(dataName1)
gp5 = data1d(dataName2)
gp7 = data1d(dataName3)



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
fig = plt.figure(figsize=(4, 3), dpi=600)

# add a subplot to the figure 'fig'.
# 1, 1, 1 means, in 1x1 grid, the first subplot.
ax = fig.add_subplot(1, 1, 1)

# plot function in matplotlib draw a line with given data.
# below example, d1.x becomes a horizontal axis,
# and d1.rho becomes the vertical axis.
ax.plot(gp3.x, gp3.rho, '-r', marker='^',linewidth=0.5, markersize=1.5)
ax.plot(gp5.x, gp5.rho, '-b', marker='s',linewidth=0.5, markersize=1.5)
ax.plot(gp7.x, gp7.rho, '-g', marker='o',linewidth=0.5, markersize=1.5)
#ax.plot(dref.x, gp7.rho, color='black', linewidth=1.0)

ax.set(xlim=(2.7, 3.5), ylim=(0., .8))


plt.ylabel(r'$\rho$')
plt.xlabel(r'$x$')

ax.legend([r'  GP-MOOD, R=1, $\rho$', r'POL-MOOD, R=1, $\rho$' ,  'Reference'], fontsize = 8)





# in order to open a graphical window
# to see the results, you need the following line

# to save a figure, use 'savefig' method under the 'fig'.
# I recommend to use a vector format
# in this case I usued 'pdf'

plt.tight_layout()

fig.savefig(plotName)
