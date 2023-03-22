from NN import *
import os

#Load the NN

lenght_list=[90,100,110]

for lenght in lenght_list:
    NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    #NN=radius_picker(max_radius=1, nb_layers=2, layer_sizes=[lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    #NN=radius_picker(max_radius=1, nb_layers=4, layer_sizes=[lenght,lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)

    NN.load('model_rot_'+str(lenght)+'.pt')

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
