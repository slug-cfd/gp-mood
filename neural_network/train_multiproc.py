import multiprocessing
from multiprocessing import current_process
from NN import *
from psutil import cpu_count

def train(lenght, dataset_file, model_name):
    #Training / Testing percentage ratio
    train_ratio=0.85
    #Batch size for training
    batch_size=1024
    #Initial learning
    lr0=0.01
    #Final learning rate
    lrend=0.0001
    #Amout of lr reduction
    max_reduction=100
    max_epoch=99999
    #Stall criterion when the lr decreases
    stall_criterion=3
    #Geometrical progression of the learning rate
    k=np.power(lrend/lr0,1.0/max_reduction) 

    #Load the dataset
    dataset = MyDataset(dataset_file)

    #define training and testing sizes
    training_size = int(len(dataset) * train_ratio)
    testing_size = len(dataset) - training_size
    
    print('DATASET FILE=', dataset_file)
    print('model_name=', model_name)
    print('DATASET SIZE=', len(dataset))
    print('TRAINING SIZE=', training_size)
    print('TESTING SIZE=', testing_size)
    print('Batch size=',batch_size)
    nbatchs=int(training_size/batch_size)
    print('nbatch=training size/batch_size=',nbatchs)

    #define training and testing datasets
    training_set, testing_set = torch.utils.data.random_split(dataset, [training_size, testing_size])

    #Shuffle the training dataset
    batched_training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    #Ordered datasets for evaluation
    testing_loader  =  DataLoader(dataset=testing_set , batch_size=testing_size, shuffle=False)

    #Define the NN, loss function and optimize
    NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght])

    loss_func = nn.MSELoss()  
    optimizer = optim.Adam(NN.parameters(), lr = lr0)

    training_loss_list=[]
    testing_loss_list=[]
    epoch_list=[]
    all_training_errors=[]
    lr_list=[]

    training_min_loss=999
    testing_min_loss=999

    lr=lr0
    nstall=0
    nreduction=0
    epoch=0

    #training loop
    while((max_reduction>nreduction)and(max_epoch>epoch)):

        #Do one epoch
        for batch_idx, (data, labels) in enumerate(batched_training_loader):
            
            #Zero gradients
            optimizer.zero_grad()
            #Compute output
            outputs=NN.forward(data)
            #Compute loss
            loss = loss_func(outputs, labels)
            #Backprop
            loss.backward()
            #step optimizer
            optimizer.step()
            #store batch error
            all_training_errors.append(loss)
     
        #Running avg of loss
        running_training_loss=sum(all_training_errors[-nbatchs:])/nbatchs
        training_loss=running_training_loss

        #count if stalling (we want 1% progress at least)
        if (training_loss.item()<training_min_loss*0.99):
            nstall=0
        else:
            nstall+=1

        #If its stalling, reduce lr
        if (nstall == stall_criterion):
            print("Lenght = ",colors.green+str(lenght)+colors.ENDC,'stalling, reducing lr from', format(lr), 'to', format(lr*k), "epoch=",str(epoch)+'/'+str(max_epoch), "nreduction=",str(nreduction+1)+'/'+str(max_reduction+1))
            print("Last training error =", format(training_loss_list[-1]))
            print("Last testing error =", format(testing_loss_list[-1]))

            #Update the learning rate
            lr=lr*k
            for g in optimizer.param_groups:
                g['lr'] = lr
                 
            nstall=0
            nreduction+=1
        
        #Every 10 epoch, dump and plot  the losses and epoch lists
        if ((epoch%3==0)and(epoch>1)):

            #Compute testing error
            for i, (data, label) in enumerate(testing_loader):
                output=NN.forward(data)
                testing_loss = loss_func(output, label)

            #if progress, save !
            if (training_loss.item()<training_min_loss):
                NN.save('model_'+model_name+'_'+str(lenght)+'.pt')
                training_min_loss=training_loss.item()
                testing_min_loss=testing_loss.item()

            #Append the losses as a function of the epochs
            training_loss_list.append(training_min_loss)
            testing_loss_list.append(testing_min_loss)
            epoch_list.append(epoch)
            lr_list.append(lr)

            #with open('losses_'+model_name+'_epoch_L_'+str(lenght)+'.pkl', 'wb') as f:
            #    pickle.dump((epoch_list, training_loss_list, testing_loss_list), f)
                
            plot_loss(epoch_list, lr_list, training_loss_list, testing_loss_list, lenght, model_name)
        
        epoch+=1
    
    #Return the last losses

    #Print some info
    print(colors.yellow+"Training for lenght= "+colors.green+str(lenght)+colors.yellow+" over !"+colors.ENDC)
    
    if (epoch>=max_epoch):
        reason=colors.yellow+'max epoch reached'+colors.ENDC
    elif (nreduction>=max_reduction):
        reason=colors.yellow+'stalled at the minimal value of lr'+colors.ENDC
    else:
        reason=colors.red+'ERROR, unknown'+colors.ENDC

    print("Reason:", reason)

    return model_name, lenght, training_loss_list[-1], testing_loss_list[-1]

if __name__ == '__main__':

    #Amount of core to share the training
    ncores=cpu_count(logical=False)
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study
    #param_list=[(k, '../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5', 'expert_2DRP3') for k in range(20,121,20)]
    param_list=[(k, '../gp-mood-code/dataset_output_2DRP3_GP_MOOD_CFL_0.8_256_256.h5', 'expert_2DRP3') for k in range(10,51,10)]

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
    
