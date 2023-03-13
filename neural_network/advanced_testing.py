from NN import *
import random

def eval_practice():
        
    NOK=0
    N_too_sharp=0
    N_too_smooth=0
    Ntot=0

    outputs=NN.forward(inputs)
    for i in range(size_testing_set):
        rpred=torch.argmax(outputs[i])
        rtarget=torch.argmax(labels[i])

        if (rpred==rtarget):
            NOK+=1
        if (rpred>rtarget):
            N_too_sharp+=1
        if (rpred<rtarget):
            N_too_smooth+=1
        Ntot+=1
    print("NOK",(100./Ntot)*NOK,"N_too_sharp",(100./Ntot)*N_too_sharp,"N_too_smooth",(100./Ntot)*N_too_smooth)
    

if (len(sys.argv)<2):
    print("error, usage: python3 train.py name")
    sys.exit()

name=sys.argv[1]
folder='sets_'+name
NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[L,L], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
NN.load(folder+'/model.pt')

print("training=")
inputs=torch.load('training_inputs.pt')
labels=torch.load('training_labels.pt')
size_testing_set=inputs.shape[0]
eval_practice()

print("testing=")
inputs=torch.load('testing_inputs.pt')
labels=torch.load('testing_labels.pt')
size_testing_set=inputs.shape[0]
eval_practice()


    



