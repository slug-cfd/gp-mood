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

file_list=sys.argv[1:]
print(file_list)

for file in file_list:

    f = h5py.File(file)

    #Get data
    time = np.array(f['time'])
    pct_detected_cell = np.array(f['pct_detected_cell'])
    steps_NN_produced_NAN= np.array(f['steps_NN_produced_NAN'])
    method=sum_char(f["method"])
    problem=sum_char(f["problem"][:])

    #Compute running avg
    running_avg_pct_detected_cell = []
    n_avg=10 #Size of the avg window

    for k in range (0, np.size(time)):
        if (k<n_avg):
            running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[0:n_avg]))
        else:
            running_avg_pct_detected_cell.append(np.mean(pct_detected_cell[k-n_avg:k]))
    plt.plot(time,running_avg_pct_detected_cell, label=method, zorder=0)



plt.xlabel('time')
plt.ylabel('pct detected cells')
plt.legend()
plt.title('compare diagnostic '+problem)
plt.savefig('compare_diagnostic_'+problem+'.png',figsize=(8, 6), dpi=100)
plt.close()