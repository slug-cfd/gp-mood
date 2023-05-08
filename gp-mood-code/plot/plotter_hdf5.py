import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

def sum_char(input):
    r=''
    for char in input:
        r=r+str(char)[-2]
    return r

file = sys.argv[1]
var  = sys.argv[2]

f = h5py.File(file)

colmap='inferno'

qqt   = np.array(f[var])

CFL=sum_char(f["CFL"][:])
method=sum_char(f["method"][:])
problem=sum_char(f["problem"][:])
lf=str(f["lf"][0])
nf=str(f["nf"][0])

title="\n problem: "+problem+" "+"CFL="+CFL+" (lf,nf)=("+lf+","+nf+")\n"+"method: "+method+"\n var: "+var

fig, (axAR) = plt.subplots(1, 1)

nx= qqt.shape[0]
ny= qqt.shape[1]

ratio = nx/ny
img = axAR.imshow(qqt, interpolation='none',cmap=colmap, extent=[0.5, -0.5, -0.5*ratio, 0.5*ratio])
ticks = np.linspace(qqt.min(), qqt.max(), 5, endpoint=True)
axAR.invert_yaxis()
plt.colorbar(img,ticks=ticks, label=var)
axAR.set_title(title, fontsize=10)
plt.savefig(file[:-3]+'_'+var+'.png', dpi=100)
fig.clf()
plt.close()
err='{:.3E}'.format(np.sum(np.abs(qqt-qqt.transpose()))/(nx*ny))
print('*----------'+var+' '+method+' '+err+'---------*')

with open("results_"+problem+".txt", "a") as myfile:

    myfile.write(method+" "+str(err)+"\n")

myfile.close()