from NN import *
from utils import *
import threading

#Training / Testing percentage ratio
train_ratio=0.85
#Batch size for training
batch_size=256
#Initial learning
lr0=0.1
#Final learning rate
lrend=lr0/1000
#AMout of epochs
num_epochs=200
#Stall criterion whe NN stops progressing
stall_criterion=50
#Amount of threads to share the training
nthreads=4
#List if NN lenght we want to study
lenght_list=range(16,256+1,16)
#step for the threading
thread_step=int(len(lenght_list)/nthreads)
#Geometrical progression of the learning rate
k=np.power(lrend/lr0,1.0/num_epochs) 

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

#train function with NN lenght as an input and threadID
def train(lenght, thread_id):

    #Define the NN, loss function and optimize
    NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    loss_func = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(NN.parameters(), lr = lr0)

    training_loss_list=[]
    testing_loss_list=[]
    epoch_list=[]

    training_min_loss=999
    testing_min_loss=999
    lr=lr0

    nstall=0
    
    #training loop
    for epoch in range(num_epochs):

        #Backpropagation
        for batch_idx, (data, labels) in enumerate(batched_training_loader):
            
            optimizer.zero_grad()

            outputs=NN.forward(data)
            loss = loss_func(outputs, labels)
            loss.backward()

            optimizer.step()
        
        #evaluation and saving if progress
        #This part should no touch the gradients
        with torch.no_grad():

            training_loss=0.0
            for i, (data, label) in enumerate(training_loader):
                output=NN.forward(data)
                training_loss += loss_func(output, label)
            
            testing_loss=0.0
            for i, (data, label) in enumerate(testing_loader):
                output=NN.forward(data)
                testing_loss += loss_func(output, label)

            print("thread_id=",thread_id, epoch*100/num_epochs,"%, training_min_loss=",training_min_loss,"testing_min_loss=",testing_min_loss,"lr=", lr)
            
            #if progress, save !
            if (training_loss.item()<training_min_loss):
                NN.save('model_'+str(lenght)+'.pt')
                training_min_loss=training_loss.item()
                testing_min_loss=testing_loss.item()
                nstall=0
            else:
                #Count if its stalling
                nstall+=1

            #Append the losses as a function of the epochs
            training_loss_list.append(training_min_loss)
            testing_loss_list.append(testing_min_loss)
            epoch_list.append(epoch)

            #Update the learning rate
            lr=lr*k

            for g in optimizer.param_groups:
                g['lr'] = lr
        
        #Every 10 epoch, dump the losses and epoch lists
        if ((epoch%10==0)and(epoch>1)):

            with open('losses_epoch_L_'+str(lenght)+'.pkl', 'wb') as f:
                pickle.dump((epoch_list, training_loss_list, testing_loss_list), f)

        #If its stalling, stop the training
        if (nstall == stall_criterion):
            print('stalling, exiting ...')
            break 
    
    #Rturn the last losses
    return training_loss_list[-1], testing_loss_list[-1]

#The function called by each thread
def my_function(thread_id):

    #define the NN lenght it will study
    thread_lenght_list=lenght_list[thread_id*thread_step:(thread_id+1)*thread_step]
    print(f"Running thread {thread_id}... with lenghts:",thread_lenght_list)

    l_l=[]
    tr_loss_list=[]
    te_loss_list=[]

    #For each NN lenght
    for lenght in thread_lenght_list:
        #train the NN and get the last losses
        tr_loss, te_loss= train(lenght, thread_id)
        l_l.append(lenght)
        tr_loss_list.append(tr_loss)
        te_loss_list.append(te_loss)

        #Dump the lenght and losses lists
        with open('losses_thread_'+str(thread_id)+'.pkl', 'wb') as f:
            pickle.dump((l_l, tr_loss_list, te_loss_list), f)


# Create a list of thread objects
threads = []

# Create and start nthreads threads
for i in range(nthreads):
    thread = threading.Thread(target=my_function, args=(i,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads finished.")