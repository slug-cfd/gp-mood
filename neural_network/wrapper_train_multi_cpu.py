from train_function import train
from utils import *

if __name__ == '__main__':

    problem=sys.argv[1]
    resolution = sys.argv[3]+"_"+sys.argv[3]
    ntrain=int(sys.argv[2])
    #Amount of core to share the training
    ncores=cpu_count(logical=False)
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study

    dataset="../gp-mood-code/dataset_output_"+problem+"_GP_MOOD_CFL_0.8_"+resolution+".h5"
    param_list=[]

    for k in range(0, ntrain+1):
        param_list.append((5, dataset, problem+"_first_10%_CEL_dropout_0.1_"+str(k), "CEL", 0.1))
 
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
    
