from NN import *
import os

#Load the NN

lenght_list=range(20,181,20)
PI_layer=False
for lenght in lenght_list:
    NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght], PI_layer=PI_layer)

    NN.load('model_no_PI_'+str(lenght)+'.pt')
    #NN.load('wrong_norm/test_rot_50_300_2_layers_85%/model_rot_100.pt')
    #Gets all the data from data/ and test each dataset separately

    data_dict = torch.load('dataset.pt')
    dataset = MyDataset(data_dict)
    size=int(len(dataset))
    dataset = DataLoader(dataset=dataset, batch_size=size, shuffle=False)

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
