import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ekin_3 = np.loadtxt('DblMach800-800_GP3rd_2quad_RK3_cfl0.8_HLL_600_ekin.data')
ekin_5 = np.loadtxt('DblMach800-800_GP5th_2quad_RK3_cfl0.8_HLL_600_ekin.data')
ekin_7 = np.loadtxt('DblMach800-800_GP7th_2quad_RK3_cfl0.8_HLL_600_ekin.data')
ekin_flash = np.loadtxt('dblJetCollisionFlash.dat')
    
fig = plt.figure(figsize=(4, 3), dpi=400)
plt.plot(ekin_3[:,0],ekin_3[:,1],'-r', marker='^',linewidth=0.5, markersize=3.5)
plt.plot(ekin_5[:,0],ekin_5[:,1],'-g', marker='x',linewidth=0.5, markersize=3.5)
plt.plot(ekin_7[:,0],ekin_7[:,1],'-b', marker='h',linewidth=0.5, markersize=3.5)
plt.plot(ekin_flash[:,0],ekin_flash[:,6],'-b', linewidth=0.5, markersize=3.5)

plt.grid()
plt.ylabel(r'Total kinetic energy ($10^4$ ergs)')
plt.xlabel(r'Time ($10^{-3}$sec)')
plt.xticks([0.001, 0.002, 0.003, 0.004, 0.005, 0.006],[r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
plt.yticks([5.E4, 10.E4, 15.E4, 20.E4],[r'$5$',r'$10$',r'$15$',r'$20$'])
plt.legend([r'3rd-order, $R=1$', r'5th-order, $R=2$', r'7th-order, $R=3$'], fontsize = 8)
plt.tight_layout()
plt.show()
fig.savefig('machJet800_TotalEkin.png')


fig2 = plt.figure(figsize=(4, 3), dpi=400)
plt.plot(ekin_3[:,0],ekin_3[:,3],'-r', marker='^',linewidth=0.5, markersize=3.5)
plt.plot(ekin_5[:,0],ekin_5[:,3],'-g', marker='x',linewidth=0.5, markersize=3.5)
plt.plot(ekin_7[:,0],ekin_7[:,3],'-b', marker='h',linewidth=0.5, markersize=3.5)
plt.grid()
#plt.ylabel(r'Peak kinetic energy ($10^4$ ergs)')
plt.xlabel(r'Time ($10^{-3}$sec)')
plt.xticks([0.001, 0.002, 0.003, 0.004, 0.005, 0.006],[r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$'])
#plt.yticks([5.E4, 10.E4, 15.E4, 20.E4],[r'$5$',r'$10$',r'$15$',r'$20$'])
plt.legend([r'3rd-order, $R=1$', r'5th-order, $R=2$', r'7th-order, $R=3$'], fontsize = 8)
plt.tight_layout()
plt.show()
fig2.savefig('machJet800_PeakEkin.png')
