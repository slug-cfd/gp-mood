import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

file = sys.argv[1]
f = h5py.File(file)

time = np.array(f['time'])
pct_detected_cell = np.array(f['pct_detected_cell'])

running_avg_pct_detected_cell = []
n_avg=5
for k in range (n_avg, np.size(time)):
    print(k)
    running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[k-5:k]))

plt.plot(time,         pct_detected_cell, color='red', label='all iteration')
plt.plot(time[n_avg:],running_avg_pct_detected_cell, color='blue', label='running avg')

plt.xlabel('time')
plt.ylabel('pct detected cells')
plt.title(file[:-3])
plt.legend()
plt.savefig(file[:-3]+'.png',figsize=(8, 6), dpi=100)
plt.close()
