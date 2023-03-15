import multiprocessing
from multiprocessing import current_process
from NN import *
from psutil import cpu_count

def train(lenght):
    #Training / Testing percentage ratio
    train_ratio=0.85
    #Batch size for training
    batch_size=512
    #Initial learning
    lr0=0.1
    #Final learning rate
    lrend=0.00001
    #Amout of lr reduction
    max_reduction=200
    max_epoch=1000
    #Stall criterion when the lr decreases
    stall_criterion=10
    #Geometrical progression of the learning rate
    k=np.power(lrend/lr0,1.0/max_reduction) 

    #Load the dataset
    data_dict = torch.load('dataset.pt')
    dataset = MyDataset(data_dict)

    #define training and testing sizes
    training_size = int(len(dataset) * train_ratio)
    testing_size = len(dataset) - training_size

    #define training and testing datasets
    training_set, testing_set = torch.utils.data.random_split(dataset, [training_size, testing_size])

    #Shuffle the training dataset
    batched_training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    #Ordered datasets for evaluation
    training_loader =  DataLoader(dataset=training_set, batch_size=training_size, shuffle=False)
    testing_loader  =  DataLoader(dataset=testing_set , batch_size=testing_size, shuffle=False)

    #Define the NN, loss function and optimize
    NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    loss_func = nn.CrossEntropyLoss()  
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

        #Backpropagation
        for batch_idx, (data, labels) in enumerate(batched_training_loader):
            
            optimizer.zero_grad()

            outputs=NN.forward(data)
            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()

            all_training_errors.append(loss)

        #evaluation and saving if progress, this part should no touch the gradients
        with torch.no_grad():

            training_loss=0.0
            # for i, (data, label) in enumerate(training_loader):
            #     output=NN.forward(data)
            #     training_loss += loss_func(output, label)

            running_training_loss=sum(all_training_errors[-batch_idx:])/batch_idx
            training_loss=running_training_loss
            #print(training_loss.item(), running_training_loss)
            
            testing_loss=0.0
            for i, (data, label) in enumerate(testing_loader):
                output=NN.forward(data)
                testing_loss += loss_func(output, label)

            #print("Lenght=",lenght,"epoch=", epoch, "training_min_loss=",format(training_min_loss),"testing_min_loss=",format(testing_min_loss),"lr=", format(lr))
            
            #if progress, save !
            if (training_loss.item()<training_min_loss):
                NN.save('model_'+str(lenght)+'.pt')
                training_min_loss=training_loss.item()
                testing_min_loss=testing_loss.item()


            #count if stalling (we want 1% progress at least)
            if (training_loss.item()<training_min_loss*0.99):
                nstall=0
            else:
                nstall+=1

            #Append the losses as a function of the epochs
            training_loss_list.append(training_min_loss)
            testing_loss_list.append(testing_min_loss)
            epoch_list.append(epoch)
            lr_list.append(lr)
        
        #Every 10 epoch, dump and plot  the losses and epoch lists
        if ((epoch%10==0)and(epoch>1)):

            with open('losses_epoch_L_'+str(lenght)+'.pkl', 'wb') as f:
                pickle.dump((epoch_list, training_loss_list, testing_loss_list), f)
            
            plot_loss(epoch_list, lr_list, training_loss_list, testing_loss_list, lenght)

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
        
        epoch+=1
    
    #Return the last losses

    #Print some info
    print(colors.yellow+"Training for lenght= "+colors.green+str(lenght)+colors.yellow+" over !"+colors.ENDC)
    
    if (epoch>=max_epoch):
        reason=colors.yellow+'max epoch reached'+colors.ENDC
    elif (lr<lrend):
        reason=colors.yellow+'stalled at the minimal value of lr'+colors.ENDC
    else:
        reason=colors.red+'ERROR, unknown'+colors.ENDC

    print("Reason:", reason)

    return lenght, training_loss_list[-1], testing_loss_list[-1]

if __name__ == '__main__':

    #Amount of core to share the training
    #ncore=cpu_count(logical=True)
    ncores=8
    print(colors.HEADER+' == Initializing the hyperparameter study on'+colors.green, ncores, colors.HEADER+'cores =='+colors.ENDC)
    #List if NN lenght we want to study
    lenght_list=range(10,80+1,10)
    print("List of hyperparameters to be shared: ", [i for i in lenght_list], "i.e"+colors.green, len(lenght_list),colors.ENDC,'elementss')

    # create a pool of processes
    pool = multiprocessing.Pool(ncores)
    
    # apply the function to the list of numbers using multiple processes
    results = pool.map(train, lenght_list)

    # close the pool
    pool.close()
    
    # print the results
    print(results)

    with open('study_L.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    for i in range(len(results)):
        result=results[i]
        if (i==0):
            hide=''
        else:
            hide='_'
        plt.scatter(result[0], result[1], color='red' , label=hide+'training_loss')
        plt.scatter(result[0], result[2], color='blue', label=hide+'testing_loss')

    plt.xlabel("lenght")
    plt.ylabel("losses")
    plt.legend()
    plt.savefig('L_study.png')
    plt.cla()
    plt.clf()
    
