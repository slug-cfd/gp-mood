import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from colors import *

torch.set_default_dtype(torch.float32) #single precision

#input types
raw_VF_data=0
MUSCL_slope=1
GP_coefficients=2

#Nb of variables for hydro simulation
n_var_hydro_2D=4

class radius_picker(nn.Module):

    def __init__(self, max_radius, nb_layers, layer_sizes, input_type=raw_VF_data, n_var_used=n_var_hydro_2D):

        super(radius_picker, self).__init__()

        print("\n")

        print("Initializing a radius picker neural network")
        self.max_radius=max_radius
        print("max radius =", max_radius)

        if (input_type==raw_VF_data):
            if (max_radius == 1 ):
                self.stencil_size = 5 #3rd order
            elif (max_radius == 2 ):
                self.stencil_size = 13 #5th order
            elif (max_radius == 3 ):
                self.stencil_size = 25 #7th order
            else:
                print(colors.red+"not implemented yet #0")
                sys.exit()
        elif(input_type==MUSCL_slope):
            print(colors.red+"not implemented yet #1")
            sys.exit()

        elif(input_type==GP_coefficients):
            print(colors.red+"not implemented yet #2")
            sys.exit()

        print("stencil size =", self.stencil_size)

        self.input_size=self.stencil_size*n_var_used
                
        print("input size =", self.input_size, "i.e. looking at", n_var_used, "var per cells")

        self.output_size=max_radius+1

        print("output size =", self.output_size, "i.e choosing between radiuses", *range(0, max_radius+1))

        if (len(layer_sizes)!=nb_layers-1):
            print(colors.red+"Error, expected", nb_layers-1, "layer sizes, got", len(layer_sizes))
            sys.exit()

        self.layers=nn.ModuleList()
        self.nb_layers=nb_layers
        
        print("initializing", nb_layers,"layers")
        self.layers.append(  nn.Linear(self.input_size, layer_sizes[0])  )
        k_layer=0
        print(k_layer, "th layer has I/O size:", self.input_size, layer_sizes[0])
        for k_layer in range(1,nb_layers-1):
            
            self.layers.append(  nn.Linear(layer_sizes[k_layer-1], layer_sizes[k_layer])  )
            print(k_layer, "th layer has I/O size:", layer_sizes[k_layer-1], layer_sizes[k_layer])

        self.layers.append(  nn.Linear(layer_sizes[nb_layers-2], self.output_size)  )
        k_layer=nb_layers-1

        print(k_layer, "th layer has I/O size:", layer_sizes[nb_layers-2], self.output_size)

        # Count the number of parameters
        num_params = sum(p.numel() for p in self.parameters())

        # Print the number of parameters
        print(colors.yellow+f"Number of parameters: {num_params}"+colors.ENDC)

        print("\n")

    def forward(self, x): #Feed forward function

        for k_layer in range(0, self.nb_layers):
            x=self.layers[k_layer](x) #layer evaluation

            if (k_layer<self.nb_layers-1): #RELU except for last layer
                x = torch.relu(x)

        return F.softmax(x,dim=1) #Softmax for last layer to output probability
    
    def pick_radius(self, x, index): #Pick a radius from data
        
        R = self.forward(x)
        R = R[index]
        p = max(R)
        R = torch.argmax(R)

        return R.item(), p.item()




NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[40,10], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)

batch_size=2
data=torch.rand((batch_size, NN.input_size)) #some random data

print(NN.forward(data)) #print the result of the NN of that data

R,p=NN.pick_radius(data, 1) #Pick a radius for the #1 data of the batch
print("Radius", R , "picked with probability", p)