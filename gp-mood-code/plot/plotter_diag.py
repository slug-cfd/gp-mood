import h5py
import sys 
import numpy as np
import matplotlib.pyplot as plt

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
niter= np.array(f['niter'])
iter_0= np.array(f['iter_0'])
niter=niter-iter_0
pct_detected_cell = np.array(f['pct_detected_cell'])
pct_detected_cell_a_priori = np.array(f['pct_detected_cell_a_priori'])
pct_detected_cell_a_posteriori = np.array(f['pct_detected_cell_a_posteriori'])


#steps_NN_produced_NAN= np.array(f['steps_NN_produced_NAN'])
#count_steps_NN_produced_NAN= f['count_steps_NN_produced_NAN'][0]
#steps_NN_sample= np.array(f['steps_NN_sample'])

problem = f['problem']

#Compute running avg
running_avg_pct_detected_cell = compute_running_average(pct_detected_cell, window_size=window_size)
running_avg_pct_detected_cell_a_priori = compute_running_average(pct_detected_cell_a_priori, window_size=window_size)
running_avg_pct_detected_cell_a_posteriori = compute_running_average(pct_detected_cell_a_posteriori, window_size=window_size)

#List of iter where the NN created NAN
# L=[]
# time_L=[]

# if sum_char(f["method"][0:10]) == "NN_GP_MOOD" :
#     color='red'
#     label='NAN states produced'
#     for niter, step in enumerate(steps_NN_produced_NAN):
#         if (step==1):
#             L.append(running_avg_pct_detected_cell[niter])
#             time_L.append(time[niter])
# elif sum_char(f["method"][0:10]) == "GP_MOOD" :
#     color='green'
#     label='dataset sampling'
#     for niter, step in enumerate(steps_NN_sample):
#         if (step==1):
#             L.append(running_avg_pct_detected_cell[niter])
#             time_L.append(time[niter])
#     print(steps_NN_sample)

#Plot detected cells

#plt.plot(time,         pct_detected_cell, color='red', label='all iteration')
plt.plot(time,running_avg_pct_detected_cell, color='blue', label='Running avg pct detected cell', zorder=0)
plt.plot(time,running_avg_pct_detected_cell_a_posteriori, color='red', label='Running avg pct detected cell a posteriori', zorder=0)
plt.plot(time,running_avg_pct_detected_cell_a_priori, color='green', label='Running avg pct detected cell a priori', zorder=0)

# #Plor iter where NN created NAN
# plt.scatter(time_L, L, color=color, marker='.', label=label, zorder=1)
# for t in time_L:
#     plt.vlines(t, np.min(running_avg_pct_detected_cell), np.max(running_avg_pct_detected_cell), colors=color, linestyles='dotted', linewidth=0.5, zorder=-1)

#axis legend
plt.xlabel('time')
plt.ylabel('pct detected cells')

#if NN GP MOOD, title with name of model, and % on steps with detected cells
# tail=''
# if sum_char(f["method"][0:10]) == "NN_GP_MOOD" :
#     n_RK=3
#     pct=(100*count_steps_NN_produced_NAN)/(n_RK*niter)
#     tail='\n count_steps_NN_produced_NAN='+str(count_steps_NN_produced_NAN)+' that is '+format(pct)+' % '
# plt.title(file[18:-3]+tail)

plt.title(file[18:-3])

plt.legend()
plt.savefig(file[:-3]+'.png', dpi=100)
plt.close()

#print(file[:-3]," ",format(pct))

#Open the file "results.txt" and write the results
with open("results_"+sum_char(problem)+".txt", "a") as myfile:

    myfile.write(sum_char(f["method"])+" "+str(np.sum(pct_detected_cell_a_posteriori/niter))+"\n")

myfile.close()


