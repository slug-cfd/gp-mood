from utils import *

class radius_picker(nn.Module):

    def __init__(self, max_radius, nb_layers, hidden_layer_sizes, softmax=False, dropout=0):
        self.init_parameters = max_radius, nb_layers, hidden_layer_sizes, softmax

        super(radius_picker, self).__init__()

        self.max_radius=max_radius

        self.softmax=softmax

        if (max_radius == 1):
            self.stencil_size = 13 #3rd order
        else:
            print("Error, not programmed")
            sys.exit()
        
        #Input size = stencil * nbvar + nbvar normalisation factors + CFL
        self.input_size=self.stencil_size*nbvar + nbvar + 1 
        #Output size = max radius + 1
        self.output_size=max_radius+1 

        if (len(hidden_layer_sizes)!=nb_layers-2):
            print(colors.red+"Error, expected", nb_layers-2, "layer sizes, got", len(hidden_layer_sizes))
            sys.exit()

        self.layers=nn.ModuleList()
        self.nb_layers=nb_layers
    
        self.layers.append(  nn.Linear(self.input_size, hidden_layer_sizes[0])  )

        for k_layer in range(1,nb_layers-2):
            
            self.layers.append(  nn.Linear(hidden_layer_sizes[k_layer-1], hidden_layer_sizes[k_layer])  )

        self.layers.append(  nn.Linear(hidden_layer_sizes[-1], self.output_size)  )

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, verbose=False): 
    #Feed forward function
        for k_layer,layer in enumerate(self.layers):
            
            #layer evaluation
            if (verbose):
                print(k_layer,"input_size = ",x.shape)

            x=layer(x) 
            x=self.dropout(x)

            if (verbose):
                print(k_layer,"output_size = ",x.shape)

            if (k_layer<self.nb_layers-2): 
                #Sigmoid except for last layer
                x=torch.sigmoid(x)
                
                if (verbose):
                    print(k_layer, "sigmoid")

        if (self.softmax):
            return torch.softmax(x,dim=1)
        else:
            return x
        
    def save(self, file='NN.pt'):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file))

def unitary_check_NN_class():

    NN1=radius_picker(max_radius=1, nb_layers=4, hidden_layer_sizes=[60,60])

    batch_size=2
    data=torch.rand((batch_size, NN1.input_size)) #some random data
    out1=NN1.forward(data, verbose=True)
    print("NN output =", out1) #print the result of the NN of that data

    #testing save and load
    NN1.save("backup.pt") #saving NN
    NN2=radius_picker(*NN1.init_parameters) #Declaring a new NN with the same parameter
    print("NN2 output before loading NN1 =", NN2.forward(data)) 
    NN2.load("backup.pt")
    out2=NN2.forward(data)
    print("NN2 output after loading NN1 = ",out2, "should be the same as NN1")
    print("difference is", torch.max(out1-out2).item())

#unitary_check_NN_class()