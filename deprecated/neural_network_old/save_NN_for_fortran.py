from utils import *
from NN import *

file=sys.argv[1]
lenght = int(sys.argv[2])
PI_layer=False
NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght], PI_layer=PI_layer)

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