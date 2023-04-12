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
f = h5py.File(file)

time = np.array(f['time'])
pct_detected_cell = np.array(f['pct_detected_cell'])
running_avg_pct_detected_cell = []
n_avg=10
for k in range (n_avg, np.size(time)):
    running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[k-n_avg:k]))

plt.plot(time,         pct_detected_cell, color='red', label='all iteration')
plt.plot(time[n_avg:],running_avg_pct_detected_cell, color='blue', label='running avg')

plt.xlabel('time')
plt.ylabel('pct detected cells')

tail=''
if sum_char(f["method"][0:10]) == "NN_GP_MOOD" :
    print("yo")
    tail='\n count_steps_NN_produced_NAN='+str(f["count_steps_NN_produced_NAN"][0])
plt.title(file[18:-3]+tail)
plt.legend()
plt.savefig(file[:-3]+'.png',figsize=(8, 6), dpi=100)
plt.close()
