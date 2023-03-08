from NN import *
import random
import matplotlib.pyplot as plt
NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[120,120], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
loss_func = nn.CrossEntropyLoss()   

lr=0.001
optimizer = optim.Adam(NN.parameters(), lr = lr)   

inputs=torch.load('inputs.pt')
labels=torch.load('labels.pt')

print(labels)
L=57
size_training_set=inputs.shape[0]

Nd=200
batch_size=int(size_training_set/Nd)

def eval(minloss,lr):

    output=NN.forward(inputs)
    loss = loss_func(batch_output, batch_label)

    print("minloss=",minloss,"lr=", lr)

    if (loss.item()<minloss):
        NN.save('model.pt')
        minloss=loss.item()

    lr*=0.99

    return minloss, lr

def eval_practice():
        
    NOK=0
    N_too_sharp=0
    N_too_smooth=0

    outputs=NN.forward(inputs)
    for i in range(size_training_set):
      #  print(outputs[i], labels[i])    
        rpred=torch.argmax(outputs[i])
        rtarget=torch.argmax(labels[i])

        if (rpred==rtarget):
            NOK+=1
        if (rpred>rtarget):
            N_too_sharp+=1
        if (rpred<rtarget):
            N_too_smooth+=1
        print(i, size_training_set)
    print("NOK",NOK,"N_too_sharp",N_too_sharp,"N_too_smooth",N_too_smooth)
    


loss_list=[]
iter_list=[]

# minloss=999
# for iter in range(1000*Nd):

#     indexes = [random.randint(0, size_training_set-1) for _ in range(batch_size)]
   
#     batch_input=inputs[indexes,:]
#     batch_label=labels[indexes,:]

#     batch_output=NN.forward(batch_input)

#     optimizer.zero_grad()
#     loss = loss_func(batch_output, batch_label)
#     loss.backward()
#     optimizer.step()

#     if (iter%(Nd-1)==0):
#         minloss,lr=eval(minloss,lr, False)
#         loss_list.append(minloss)
#         iter_list.append(iter)


# plt.plot(iter_list, loss_list, label='loss')
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.semilogy(iter_list, loss_list)
# plt.legend()
# plt.savefig('progress.png')


NN.load('model.pt')
eval_practice()


    



