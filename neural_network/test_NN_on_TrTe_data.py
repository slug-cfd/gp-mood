from NN import *
import os

#Load the NN

lenght_list=range(10,51,10)
#dataset = MyDataset('expert_shu_osher/dataset_output_shu_osher_GP_MOOD_CFL_0.8_256_256.h5')

dataset = MyDataset('expert_sedov/more_sample/dataset_output_sedov_GP_MOOD_CFL_0.8_256_256.h5')

size=int(len(dataset))
dataset = DataLoader(dataset=dataset, batch_size=size, shuffle=False)

for lenght in lenght_list:
    NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght], softmax=True)

    #NN.load('model_expert_shu_osher_softmax_'+str(lenght)+'.pt')
    #NN.load("expert_shu_osher/model_expert_shu_osher_"+str(lenght)+'.pt')
   # NN.load("expert_sedov/more_sample/model_expert_sedov_"+str(lenght)+'.pt')
    NN.load("model_expert_sedov_softmax_"+str(lenght)+'.pt')

    for batch_idx, (data, labels) in enumerate(dataset):
        outputs=NN.forward(data)

    rpred = torch.argmax(outputs, dim=1)
    rtarget = torch.argmax(labels, dim=1)

    NFALSE=0
    NP=0
    NM=0
    for i in range(size):
        if (rpred[i]!=rtarget[i]):
            NFALSE+=1
            if (rpred[i]>rtarget[i]):
                NP+=1
            else:
                NM+=1

    print(lenght, NFALSE*100/size, "Too sharp:", format(NP*100/NFALSE), 'Too smooth', format(NM*100/NFALSE))
