from NN import *
import os

#Load the NN
NN=radius_picker(max_radius=1, nb_layers=3, layer_sizes=[80,80], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
NN.load('model_80.pt')

#Gets all the data from data/ and test each dataset separately

paths=[]
for file_name in os.listdir('data/'):
    paths.append('data/'+file_name)

print(paths)

for path in paths:

    problem=path[8:15]
    CFL=path[20:23]
    radius=path[-10]

    input=torch.load(path)
    outputs=NN.forward(input)

    N=input.shape[0]

    radius_int=int(radius)

    NFALSE=0

    for i in range(N):
        rpred=torch.argmax(outputs[i])
        if (rpred!=radius_int):
            NFALSE+=1

    print(problem, CFL, radius, NFALSE*100/N)
