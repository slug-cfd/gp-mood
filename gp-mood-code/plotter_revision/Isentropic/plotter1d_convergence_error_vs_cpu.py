# lets import essential packages: numpy and matplotlib.
# with nickname of np and plt.
import numpy as np
import matplotlib.pyplot as plt


error_pol3=np.array([1.054e0,2.570e-1,4.432e-2,5.944e-3])
error_gp3=np.array([1.137e+0,1.675e-1,2.917e-2,4.116e-3])
cpu_pol3=np.array([0.714, 4.5, 33.4, 252])
cpu_gp3=np.array([1.06,5.15, 39.9, 298])

fig=plt.figure(figsize=(4,3),dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.log10(cpu_pol3),np.log10(error_pol3), 'b', marker='^',linewidth=0.5, markersize=4)
ax.plot(np.log10(cpu_gp3),np.log10(error_gp3), 'r', marker='s',linewidth=0.5, markersize=4)

plt.xlabel(r'CPU Times (sec)')
plt.ylabel(r'$L_1$ error')
plt.legend([r'POL-MOOD3', r'GP-MOOD3'])
plt.tight_layout()
plt.show()

"""
fig2 = plt.figure(figsize=(4, 3), dpi=400)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(gridsLog, gp3_2_L1, 'b', marker='^',linestyle = 'None', markersize=4)
ax2.plot(gridsLog, gp5_3_L1, 'g', marker='s',linestyle = 'None', markersize=4)
ax2.plot(gridsLog, gp7_4_L1, 'r', marker='>',linestyle = 'None', markersize=4)
ax2.plot(gridsLog, pol3_2_L1, 'm', marker='v',linestyle = 'None', markersize=4)

ax2.plot(gridsLog, -np.log10(grids**3)+5.3, '--b', linewidth=0.5)
ax2.plot(gridsLog, -np.log10(grids**5)+8.2, '--g', linewidth=0.5)
ax2.plot(gridsLog, -np.log10(grids**7)+11.6, '--r', linewidth=0.5)
ax2.plot(gridsLog, -np.log10(grids**3)+5.5, '--m', linewidth=0.5)

#ax.set(xlim=(2.7, 3.5), ylim=(0., .8))


plt.xlabel(r'Grid Resolution')
plt.xticks(gridsLog,[r'$50$',r'$100$',r'$200$',r'$400$'])

plt.ylabel(r'$L_1$ error')
#plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0, 1],[r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
plt.yticks([-7, -5, -3, -1, 1],[r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$', r'$10^{1}$'])

#plt.legend([r'3rd-order GP-MOOD, $R=1$', r'5th-order GP-MOOD, $R=2$', r'7th-order GP-MOOD, $R=3$'], fontsize = 8)
#plt.legend([r'3rd-order GP-MOOD3', r'5th-order GP-MOOD5', r'7th-order GP-MOOD7'], fontsize = 8)
plt.legend([r'3rd-order GP-MOOD3', r'5th-order GP-MOOD5', r'7th-order GP-MOOD7', r'3rd-order POL-MOOD3'], fontsize = 8)
plt.tight_layout()
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
"""
