from utils import *
from NN import *

file=sys.argv[1]
lenght = int(sys.argv[2])

NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[lenght,lenght], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)

NN.load(file)

iparam=0
for param in NN.parameters():

    if (iparam==0):
        type='w'
    else:
        type='a'

    with open(file[:-3]+'_fortran.txt', type) as f:
        # Iterate over the elements of the array and write each one to a new line in the file
        for element in param.flatten():
            f.write(str(element.item()) + '\n')
            
    iparam+=1

x=torch.ones(L)
print(NN.forward(x))#, F.sigmoid(x))