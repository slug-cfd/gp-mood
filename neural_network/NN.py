import sys
from utils import *

torch.set_default_dtype(torch.float32) 

#input types
raw_VF_data=0
MUSCL_slope=1
GP_coefficients=2

#Nb of variables for hydro simulation
n_var_hydro_2D=4

class radius_picker(nn.Module):

    def __init__(self, max_radius, nb_layers, layer_sizes, input_type=raw_VF_data, n_var_used=n_var_hydro_2D):
        self.init_parameters = max_radius, nb_layers, layer_sizes, input_type, n_var_used

        super(radius_picker, self).__init__()

        self.max_radius=max_radius

        if (input_type==raw_VF_data):
            if (max_radius == 1 ):
                self.stencil_size = 13 #3rd order
            elif (max_radius == 2 ):
                self.stencil_size = 25 #5th order
            elif (max_radius == 3 ):
                self.stencil_size = -1000 #7th order
            else:
                print(colors.red+"not implemented yet #0")
                sys.exit()
        elif(input_type==MUSCL_slope):
            print(colors.red+"not implemented yet #1")
            sys.exit()

        elif(input_type==GP_coefficients):
            print(colors.red+"not implemented yet #2")
            sys.exit()

        self.input_size=self.stencil_size*n_var_used + n_var_used + 1 #(variables + normalisation factors) #+ CFL )
                
        self.output_size=max_radius+1 #R=2, output_size = 2+1=3 = {0,1,2}

        if (len(layer_sizes)!=nb_layers-1):
            print(colors.red+"Error, expected", nb_layers-1, "layer sizes, got", len(layer_sizes))
            sys.exit()

        self.layers=nn.ModuleList()
        self.nb_layers=nb_layers
        
        #self.layers.append(  nn.Linear(self.input_size, layer_sizes[0])  )
        p_table=compute_permutation_table_90_rot()
        self.layers.append( PermutationInvariantLinear(self.input_size, layer_sizes[0], p_table)  )

        for k_layer in range(1,nb_layers-1):
            
            self.layers.append(  nn.Linear(layer_sizes[k_layer-1], layer_sizes[k_layer])  )

        self.layers.append(  nn.Linear(layer_sizes[nb_layers-2], self.output_size)  )

        num_params = sum(p.numel() for p in self.parameters())

        #print('\n'+colors.HEADER+" --- Initialized a radius picker neural network ---"+colors.ENDC)
        #print("max radius =", max_radius)
        #print("stencil size =", self.stencil_size)
        #print("input size =", self.input_size, "i.e. looking at", n_var_used, "var per cells")
        #print("output size =", self.output_size, "i.e choosing between radiuses", *range(0, max_radius+1))
        #print("Initialized", nb_layers,"layers")
        #print(colors.yellow+f"Number of parameters: {num_params}"+colors.ENDC)
        #print(colors.HEADER+" ---------------------- End -----------------------\n"+colors.ENDC)

    def forward(self, x): #Feed forward function

        for k_layer in range(0, self.nb_layers):
            x=self.layers[k_layer](x) #layer evaluation

            if (k_layer<self.nb_layers-1): #RELU except for last layer
              #  x = torch.relu(x)
                x=torch.sigmoid(x)
        #return F.softmax(x,dim=1) #Softmax for last layer to output probability
        return x
    
    def pick_radius(self, x, index): #Pick a radius from data
        
        R = self.forward(x)
        R = R[index]
        p = max(R)
        R = torch.argmax(R)

        return R.item(), p.item()
    
    def save(self, file='NN.pt'):
       # print(colors.yellow+"... Saving the NN in file ...", file)
        torch.save(self.state_dict(), file)
       # print(colors.green+"Saved the NN in file", file, colors.ENDC)

    def load(self, file):
       # print(colors.yellow+"... Loading the NN from file ...", file)
        self.load_state_dict(torch.load(file))
       # print(colors.green+"Loaded the NN from file", file, colors.ENDC)

def unitary_check_NN_class():

    NN1=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[58,58], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)

    batch_size=2
    data=torch.rand((batch_size, NN1.input_size)) #some random data
    out1=NN1.forward(data)
    print("NN output =", out1) #print the result of the NN of that data

    R,p=NN1.pick_radius(data, 1) #Pick a radius for the #1 data of the batch
    print("Radius", R , "picked with probability", p)

    #testing save and load
    NN1.save("backup.pt") #saving NN
    NN2=radius_picker(*NN1.init_parameters) #Declaring a new NN with the same parameter
    print("NN2 output before loading NN1 =", NN2.forward(data)) 
    NN2.load("backup.pt")
    out2=NN2.forward(data)
    print("NN2 output after loading NN1 = ",out2, "should be the same as NN1")
    print("difference is", torch.max(out1-out2).item())