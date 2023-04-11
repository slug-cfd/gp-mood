from utils import *
from NN import *
from symmetries import *

def reflexion():

    p_table=compute_permutation_table_90_rot()

    input=torch.randn((L))
    p_input=rotate_90(input,p_table)
    p_input=rotate_90(p_input,p_table)
    p_input=rotate_90(p_input,p_table)
    p_input=rotate_90(p_input,p_table)

    print(p_input-input)
    
    p_input=rotate_90(p_input,p_table)

    Layer = PermutationInvariantLinear(L,1, p_table)
    #Layer = RotationalInvariantLinear(L,0, p_table)

    print(Layer(input))
    print(Layer(p_input))

    model = radius_picker(max_radius=1, nb_layers=3, layer_sizes=[20,20], input_type=raw_VF_data, n_var_used=n_var_hydro_2D)
    
    print(model(input))
    print(model(p_input))

    model.load('model_PI_20.pt')

    print(model(input))
    print(model(p_input))

reflexion()