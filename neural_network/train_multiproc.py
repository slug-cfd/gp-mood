import multiprocessing
from multiprocessing import current_process
from train_function import train
from psutil import cpu_count
from utils import *

if (len(sys.argv)<3):
    print("ERROR, usage: python3.9 train_multiproc dataset model_name")
    sys.exit()

dataset_file=sys.argv[1]
model_name=sys.argv[2]

if __name__ == '__main__':

    #Amount of core to share the training
    ncores=cpu_count(logical=False)-1
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study
    #param_list=[(k, '../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5', 'expert_2DRP3') for k in range(20,121,20)]
    param_list=[(k, dataset_file, model_name, True) for k in range(10,101,10)]

    print("List of hyperparameters to be shared: ", [i for i in param_list], "i.e"+colors.green, len(param_list),colors.ENDC,'elements')

    # create a pool of processes
    pool = multiprocessing.Pool(ncores)
    
    # apply the function to the list of numbers using multiple processes
    results = pool.starmap(train, param_list)

    # close the pool
    pool.close()
    
    # print the results
    print(results)
    
    for i, result in enumerate(results):
        if (i==0):
            hide=''
        else:
            hide='_'
        plt.scatter(result[1], result[2], color='red' , label=hide+'training_loss')
        plt.scatter(result[1], result[3], color='blue', label=hide+'testing_loss')

    plt.xlabel("L")
    plt.ylabel("losses")
    plt.legend()
    plt.savefig('L_study.png')
    plt.cla()
    plt.clf()
    
