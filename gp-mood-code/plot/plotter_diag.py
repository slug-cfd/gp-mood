import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

def sum_char(input):
    r=''
    for char in input:
        r=r+str(char)[-2]
    return r

def format(x):
    """Format a double with one digit after the comma"""
    return f"{x:.5f}"

#Get file
file = sys.argv[1]
f = h5py.File(file)

#Get data
time = np.array(f['time'])
pct_detected_cell = np.array(f['pct_detected_cell'])
steps_NN_produced_NAN= np.array(f['steps_NN_produced_NAN'])
count_steps_NN_produced_NAN= f['count_steps_NN_produced_NAN'][0]
steps_NN_sample= np.array(f['steps_NN_sample'])

#Compute running avg
running_avg_pct_detected_cell = []
n_avg=10 #Size of the avg window

for k in range (0, np.size(time)):
    if (k<n_avg):
        running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[0:n_avg]))
    else:
        running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[k-n_avg:k]))

#List of iter where the NN created NAN
L=[]
time_L=[]

if sum_char(f["method"][0:10]) == "NN_GP_MOOD" :
    color='red'
    label='NAN states produced'
    for niter, step in enumerate(steps_NN_produced_NAN):
        if (step==1):
            L.append(running_avg_pct_detected_cell[niter])
            time_L.append(time[niter])
elif sum_char(f["method"][0:10]) == "GP_MOOD" :
    color='green'
    label='dataset sampling'
    for niter, step in enumerate(steps_NN_sample):
        if (step==1):
            L.append(running_avg_pct_detected_cell[niter])
            time_L.append(time[niter])
    print(steps_NN_sample)

#Plot detected cells

#plt.plot(time,         pct_detected_cell, color='red', label='all iteration')
plt.plot(time,running_avg_pct_detected_cell, color='blue', label='Running avg pct detected cell', zorder=0)

#Plor iter where NN created NAN

plt.scatter(time_L, L, color=color, marker='.', label=label, zorder=1)
for t in time_L:
    plt.vlines(t, np.min(running_avg_pct_detected_cell), np.max(running_avg_pct_detected_cell), colors=color, linestyles='dotted', linewidth=0.5, zorder=-1)

#axis legend
plt.xlabel('time')
plt.ylabel('pct detected cells')

#if NN GP MOOD, title with name of model, and % on steps with detected cells
tail=''
if sum_char(f["method"][0:10]) == "NN_GP_MOOD" :
    n_RK=3
    pct=(100*count_steps_NN_produced_NAN)/(n_RK*niter)
    tail='\n count_steps_NN_produced_NAN='+str(count_steps_NN_produced_NAN)+' that is '+format(pct)+' % '
plt.title(file[18:-3]+tail)

plt.legend()
plt.savefig(file[:-3]+'.png', dpi=100)
plt.close()

print(file[:-3]," ",format(pct))