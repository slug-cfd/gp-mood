from train_function import train
from utils import *


if __name__ == '__main__':

    #Amount of core to share the training
    ncores=cpu_count(logical=False)-1
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study
    param_list=[]
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_no_dropout_0", "CEL", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_no_dropout_1", "CEL", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_no_dropout_2", "CEL", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_no_dropout_3", "CEL", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_no_dropout_4", "CEL", 0))
 
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_dropout_0", "CEL", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_dropout_1", "CEL", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_dropout_2", "CEL", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_dropout_3", "CEL", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_CEL_dropout_4", "CEL", 0.05))

    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_no_dropout_0", "MSE", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_no_dropout_1", "MSE", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_no_dropout_2", "MSE", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_no_dropout_3", "MSE", 0))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_no_dropout_4", "MSE", 0))

    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_dropout_0", "MSE", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_dropout_1", "MSE", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_dropout_2", "MSE", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_dropout_3", "MSE", 0.05))
    param_list.append((20, "../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5", "2DRP3_first_100_MSE_dropout_4", "MSE", 0.05))

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
    
