from NN import *

def train(lenght, dataset_file, model_name, softmax, gpu_id=0):
            
    device = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')

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
    stall_criterion=5
    #Geometrical progression of the learning rate
    k=np.power(lrend/lr0,1.0/max_reduction) 

    #Load the dataset
    dataset = MyDataset(dataset_file)

    #define training and testing sizes
    training_size = int(len(dataset) * train_ratio)
    testing_size = len(dataset) - training_size
    print("\n---------------------------------")
    print("\n"+colors.HEADER+"CUDA AVAILABLE:", torch.cuda.is_available(), colors.ENDC+"\n")
    print('DATASET FILE:', dataset_file)
    print('model_name:', model_name)
    print('DATASET SIZE:', len(dataset))
    print('TRAINING SIZE:', training_size)
    print('TESTING SIZE:', testing_size)
    print('Batch size:',batch_size)
    nbatchs=int(training_size/batch_size)
    print('nbatch=training size/batch_size:',nbatchs)
    print("---------------------------------\n")


    #define training and testing datasets
    training_set, testing_set = torch.utils.data.random_split(dataset, [training_size, testing_size])

    #Shuffle the training dataset
    batched_training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    #Ordered datasets for evaluation
    testing_loader  =  DataLoader(dataset=testing_set , batch_size=testing_size, shuffle=False)

    #Define the NN, loss function and optimize
    NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght], softmax=softmax).to(device)

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

    t_epochs=0

    #training loop
    t_beg_training=time.time()
    while((max_reduction>nreduction)and(max_epoch>epoch)):

        #Do one epoch

        t_beg_epoch=time.time()

        for batch_idx, (data, labels) in enumerate(batched_training_loader):
            #send data and labels to device 
            data=data.to(device)
            labels=labels.to(device)
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

        t_end_epoch=time.time()
        t_epochs+=t_end_epoch-t_beg_epoch
     
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
            print("Lenght = ",colors.green+str(lenght)+colors.ENDC,'stalling, reducing lr:', format(lr), '->', format(lr*k), "epoch:",colors.green+str(epoch)+'/'+str(max_epoch)+colors.ENDC, "nreduction:",colors.green+str(nreduction+1)+'/'+str(max_reduction+1)+colors.ENDC)
            t_now=time.time()
            total_time=t_now-t_beg_training
            rest=total_time-t_epochs
            print("Perf (epochs only  ): Time per epoch:", colors.green+format(t_epochs/epoch)  ,"s"+colors.ENDC, " Mdatapoint per s:", colors.green+format(epoch*training_size/(t_epochs*10000))+colors.ENDC)
            print("Perf (all includend): Time per epoch:", colors.green+format(total_time/epoch),"s"+colors.ENDC, " Mdatapoint per s:", colors.green+format(epoch*training_size/(total_time*10000))+colors.ENDC)
            print("Total time:", colors.green+format(total_time/60), "mins"+colors.ENDC+". Epochs time:", colors.green+format(t_epochs/60),'mins'+colors.ENDC+':', colors.green+format(100*t_epochs/total_time), "%"+colors.ENDC," Rest:", colors.green+format(rest/60),'mins'+colors.ENDC+':', colors.green+format(100*rest/total_time), "%"+colors.ENDC+'\n')
            #Update the learning rate
            lr=lr*k
            for g in optimizer.param_groups:
                g['lr'] = lr
                 
            nstall=0
            nreduction+=1
        
        #Every 10 epoch, dump and plot  the losses and epoch lists
        if ((epoch%10==0)and(epoch>1)):

            #Compute testing error
            for i, (data, label) in enumerate(testing_loader):
                data=data.to(device)
                label=label.to(device)
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

