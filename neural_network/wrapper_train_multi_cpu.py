from train_function import train
from utils import *


if __name__ == '__main__':

    #Amount of core to share the training
    ncores=cpu_count(logical=False)-1
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study
    param_list=[]
    param_list.append((10, "mixed_all_2DRP/try_2_all_dataset/mixed_all_2DRP.h5", "mixed_all_2DRP_CEL", "CEL"))
    param_list.append((20, "mixed_all_2DRP/try_2_all_dataset/mixed_all_2DRP.h5", "mixed_all_2DRP_CEL", "CEL"))
    param_list.append((30, "mixed_all_2DRP/try_2_all_dataset/mixed_all_2DRP.h5", "mixed_all_2DRP_CEL", "CEL"))
    param_list.append((40, "mixed_all_2DRP/try_2_all_dataset/mixed_all_2DRP.h5", "mixed_all_2DRP_CEL", "CEL"))
    param_list.append((50, "mixed_all_2DRP/try_2_all_dataset/mixed_all_2DRP.h5", "mixed_all_2DRP_CEL", "CEL"))

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
    
