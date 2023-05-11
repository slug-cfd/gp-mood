from utils import *
from NN import *

file=sys.argv[1]
lenght = int(file[-4:-3])
#lenght = int(file[-5:-3])
print(lenght)
NN=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[lenght,lenght], softmax=False)

NN.load(file)

iparam=0
for param in NN.parameters():

    if (iparam==0):
        type='w'
    else:
        type='a'

    with open(file[:-3]+'.txt', type) as f:
        # Iterate over the elements of the array and write each one to a new line in the file
        for element in param.flatten():
            f.write(str(element.item()) + '\n')
            
    iparam+=1

x=torch.ones((1,L))
print(NN.forward(x))