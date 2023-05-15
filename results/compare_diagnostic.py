import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys 

window_size = 30

def compute_running_average(data, window_size):
    """
    Compute running average of data
    """
    running_average = np.zeros(data.shape)
    for i in range(data.shape[0]):
        if i < window_size:
            running_average[i] = np.mean(data[:i+1])
        else:
            running_average[i] = np.mean(data[i-window_size:i+1])
    return running_average

problem=sys.argv[1]
resolution=sys.argv[2]+'_'+sys.argv[2]

# Load the GP-MOOD diagnostic data: 
file='diagnostic_output_'+problem+'_'+'GP_MOOD_CFL_0.8_'+resolution+'.h5'
print(file)
data = h5py.File(file,'r')

#Get data 
time_GP_MOOD = np.array(data['time'])
pct_detected_cell = np.array(data['pct_detected_cell'])

#Compute running avg
running_avg_GP_MOOD=compute_running_average(pct_detected_cell, window_size=window_size)

#Get all diagnostic data from NN_GP_MOOD datafile

step_min=999999
for n_NN in range(10):

    file='diagnostic_output_'+problem+'_'+'NN_GP_MOOD_CC_model_'+problem+'_first_10%_CEL_dropout_0.1_'+str(n_NN)+'_5_CFL_0.8_'+resolution+'.h5'
    data = h5py.File(file,'r')
    pct_detected_cell = np.array(data['pct_detected_cell'])
    time = np.array(data['time'])
    size=pct_detected_cell.shape[0]
    if (size < step_min):
        step_min=size
        time_NN_GP_MOOD=time
    

#Compute avg over all runs
run_avg=np.zeros(step_min)
for n_NN in range(10):

    file='diagnostic_output_'+problem+'_'+'NN_GP_MOOD_CC_model_'+problem+'_first_10%_CEL_dropout_0.1_'+str(n_NN)+'_5_CFL_0.8_'+resolution+'.h5'
    data = h5py.File(file,'r')
    pct_detected_cell = np.array(data['pct_detected_cell'])
    run_avg = run_avg + pct_detected_cell[0:step_min]

run_avg=run_avg/10

#Plot running average of detected cells
plt.plot(time_GP_MOOD[window_size:], running_avg_GP_MOOD[window_size:], color='black', label='GP-MOOD')
plt.plot(time_NN_GP_MOOD[window_size:], run_avg[window_size:], color='red', label='NN-GP-MOOD (avegared over 10 runs)')
plt.plot()
plt.xlabel('Time')
plt.ylabel('Percentage of detected cells')
plt.title(problem)
plt.legend()
plt.savefig('compare_low_order_update_'+problem+'.png', dpi=100)
