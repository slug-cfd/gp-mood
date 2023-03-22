from NN import *
import os

#Load the NN

lenght_list=[90,100,110]

for lenght in lenght_list:
    NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    #NN=radius_picker(max_radius=1, nb_layers=2, layer_sizes=[lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    #NN=radius_picker(max_radius=1, nb_layers=4, layer_sizes=[lenght,lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)

    NN.load('model_norot_'+str(lenght)+'.pt')

    #Gets all the data from data/ and test each dataset separately

    data_dict = torch.load('dataset_rot.pt')
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

    NSym=0
    NnotSym=0
    for i in range(size):

        if (i%4==0):
            symmety = True
            rm1=rpred[i+1]
        else:
            if (rm1!=rpred[i]):
                symmety = False
            if (i%4==3):
                if (symmety):
                    NSym+=1
                else:
                    NnotSym+=1

        if (rpred[i]!=rtarget[i]):
            NFALSE+=1
            if (rpred[i]>rtarget[i]):
                NP+=1
            else:
                NM+=1
        #print(i,[format(k.item()) for k in data[i]],"\n")

    print(lenght, NFALSE*100/size, "Too sharp:", format(NP*100/NFALSE), 'Too smooth', format(NM*100/NFALSE))
    print("Nsym", NSym*100/(NSym+NnotSym), "N_not_sym", NnotSym*100/(NSym+NnotSym))