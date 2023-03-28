from utils import *
base_indexes=dict()

base_indexes[( 0 ,  0 )] = 0
base_indexes[(-1,   0 )] = 1
base_indexes[( 0 ,  1 )] = 2
base_indexes[( 1 ,  0 )] = 3
base_indexes[( 0,  -1 )] = 4
base_indexes[(-1,  -1 )] = 5
base_indexes[(-1 ,  1 )] = 6
base_indexes[( 1 ,  1 )] = 7
base_indexes[( 1 , -1 )] = 8
base_indexes[( 0,  -2 )] = 9
base_indexes[(-2 ,  0 )] = 10
base_indexes[( 0 ,  2 )] = 11
base_indexes[( 2 ,  0 )] = 12

rotations=[(1,1), (-1,1), (-1,-1), (1,-1)]
rotated_indexes=[[],[],[],[]]
nrot=0
for rotation in rotations:
    for key in base_indexes.keys():
        rotated_indexes[nrot].append(base_indexes[(key[0]*rotation[0], key[1]*rotation[1])])
    nrot=nrot+1

#for rotated_index in rotated_indexes:
#    print(rotated_index)

id=0
imx=1
imy=2
iE=3
nbvar=4

index_all_mx=range(imx, L-nbvar-1, nbvar)
index_all_my=range(imy, L-nbvar-1, nbvar)

#print( "all imx", [i for i in index_all_mx] )
#print( "all imy", [i for i in index_all_my] )

def rotate(data):

    rotated_data=[torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666]
    nrot=0

    for rotation in rotations:
        
        #Rotate the FV data
        for icell in range(13):
            #print(data.shape, rotated_indexes[nrot][icell]*nbvar,rotated_indexes[nrot][icell]*nbvar+nbvar-1)
            beg_rotated_data=icell*nbvar
            beg_data=rotated_indexes[nrot][icell]*nbvar
            lenght=nbvar
            rotated_data[nrot][beg_rotated_data:beg_rotated_data+lenght]=data[beg_data:beg_data+lenght]

        #Flip the x,y speed accordingly
        rotated_data[nrot][index_all_mx]*=rotation[0]
        rotated_data[nrot][index_all_my]*=rotation[1]
        #Get the CFL
        rotated_data[nrot][-1]=data[-1]
        #Get the normalisaiton factors (they dont flip, see test_norm)
        rotated_data[nrot][-nbvar-1:-1]=data[-nbvar-1:-1]
        #rotated_data[nrot]-=data
        nrot=nrot+1

    return rotated_data

def unit_resting_rotation(data0):

    data1=rotate(data0)     # 1: 11  -11 -1-1 1-1
    data2=rotate(data1[-1]) # 2: 1-1 -1-1 -11 11
    data3=rotate(data2[-1]) # 3: 11  -11 -1-1 1-1
    data4=rotate(data3[-1]) # 4: 1-1 -1-1 -11 11

    print("2 rotations should give 0 difference:",torch.max(data1[0]-data2[3]))
    print("2 rotations should give 0 difference:",torch.max(data1[0]-data3[0]))
    print("2 rotations should give 0 difference:",torch.max(data1[0]-data4[3]))

    print("2 rotations should give 0 difference:",torch.max(data1[1]-data2[2]))
    print("2 rotations should give 0 difference:",torch.max(data1[1]-data3[1]))
    print("2 rotations should give 0 difference:",torch.max(data1[1]-data4[2]))

    print("2 rotations should give 0 difference:",torch.max(data1[2]-data2[1]))
    print("2 rotations should give 0 difference:",torch.max(data1[2]-data3[2]))
    print("2 rotations should give 0 difference:",torch.max(data1[2]-data4[1]))

    print("2 rotations should give 0 difference:",torch.max(data1[3]-data2[0]))
    print("2 rotations should give 0 difference:",torch.max(data1[3]-data3[3]))
    print("2 rotations should give 0 difference:",torch.max(data1[3]-data4[0]))

def norm(x):
    xmin=torch.min(x)
    xmax=torch.max(x)

    xnorm=(x-0.5*(xmax+xmin))*(2./(xmax-xmin))
    F=torch.sign(xmin)*(xmax-xmin)

    return xnorm, F
    
def test_norm():

    data=torch.zeros((5))

    data[0]=1
    data[1]=-9
    data[2]=7
    data[3]=5
    data[4]=-10

    print(norm( data))
    print(norm(-data))

def compute_permutation_table_90_rot():
    rot90=[]

    for key in base_indexes.keys():
        rot90.append(base_indexes[(key[1], -key[0])])

    index=torch.zeros(L)
    p_table=torch.zeros(L)

    for i in range(L):
        index[i]=i
        p_table[i]=i

    for icell in range(13):
            beg_rotated_data=icell*nbvar
            beg_data=rot90[icell]*nbvar
            lenght=nbvar
            p_table[beg_rotated_data:beg_rotated_data+lenght]=index[beg_data:beg_data+lenght]

    all_mx = p_table[index_all_mx]
    all_my = p_table[index_all_my]
    
    p_table[index_all_mx] = all_my
    p_table[index_all_my] = -all_mx

    return p_table

def rotate_90(input, p_index):

    premutted_input = torch.zeros(L)

    for i in range(L):
        signe=sign(p_index[i])
        premutted_input[i]=signe*input[int(abs(p_index[i]))]
    
    return premutted_input

def reflexion():

    p_table=compute_permutation_table_90_rot()

    input=torch.randn((L))
    p_input=rotate_90(input,p_table)

    Layer = PermutationInvariantLinear(L,1, p_table)
    #Layer = RotationalInvariantLinear(L,0, p_table)

    print(Layer(input))
    print(Layer(p_input))

class PermutationInvariantLinear(nn.Module):

    def __init__(self, in_features, out_features, permutation_table):
        super(PermutationInvariantLinear, self).__init__()

        #in_features: dimension of the input of the NN (1D)
        self.in_features=in_features

        #out_fetures: dimension of the output of the NN (1D)
        self.out_features=out_features

        #permutation_table: torch array of the same dimension than an input array that describres the permutation
        self.permutation_table=[int(abs(i.item())) for i in permutation_table]
        self.sign_table=[sign(i.item()) for i in permutation_table]
        #Define weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        #Check the size of the table
        if (len(self.permutation_table)!=in_features):
            print('Error, wrong size for the permutation table')

        self.code_zero=666

        #Computes the DOF of the NN
        self.enfore_requires_grad()
        self.enforce_symmetric_weights()

    def forward(self, x):
        
        # Enforce the condition symmetry condition the weight
        self.enforce_symmetric_weights()
        
        # Compute the output using the weights and biases
        output = torch.matmul(x, self.weight.t()) + self.bias

        return output

    def enforce_symmetric_weights(self):

        with torch.no_grad():
            for line in range(self.out_features):
                for i in range(self.in_features):
                    index=self.permutation_table[i]
                    if (index==self.code_zero):
                        self.weight[line,i]=0.0
                    else:
                        self.weight[line,i]=self.sign_table[i]*self.weight[line,index]

    def enfore_requires_grad(self):

        done=False
        niter=0
        while (not done):
            niter+=1
            done=True
            for k in range(L):

                target=self.permutation_table[self.permutation_table[k]]

                sign_target=self.sign_table[self.permutation_table[k]]
                                            
                if (self.permutation_table[k]!=target):
                    self.permutation_table[k]=target
                    self.sign_table[k]*=sign_target
                    done=False

        print(done, niter)
        print([k for k in self.permutation_table])
        print([k for k in self.sign_table])

        for k in range(L):
            if (k==self.permutation_table[k] and self.sign_table[k]<0):
                for k2, item in enumerate(self.permutation_table):
                    if (item==k):
                        self.permutation_table[k2] = self.code_zero

        with torch.no_grad():
            self.bias.requires_grad_(True)

            for line in range(self.out_features):
                for k in range(self.in_features):
                        if (k in self.permutation_table):
                            self.weight[line, k].requires_grad_(True)
                            #print(k, True)
                        else:
                            self.weight[line, k].requires_grad_(False)
                            #print(k, False)
        print(len(list(set(self.permutation_table))))


class RotationalInvariantLinear(nn.Module):

    def __init__(self, in_features, out_features, rotation_table):
        super(RotationalInvariantLinear, self).__init__()

        #in_features: dimension of the input of the NN (1D)
        #out_fetures: dimension of the output of the NN (1D)
        #rotation_table: torch array of the same dimension than an input array that describres the rotation

        self.in_features=in_features
        self.out_features=out_features

        #Define weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        #rearrange the rotation table in a list
        self.rotation_table=[int(i.item()) for i in rotation_table]
        print(self.rotation_table)
        #Check the size of the table
        if (len(self.rotation_table)!=in_features):
            print('Error, wrong size for the rotation table')

        #Computes the DOF of the NN
        self.enfore_requires_grad(self.rotation_table)


    def forward(self, x):
        
        # Enforce the condition symmetry condition the weight
        self.enforce_symmetric_weights()
        
        # Compute the output using the weights and biases
        output = torch.matmul(x, self.weight.t()) + self.bias

        return output

    def enforce_symmetric_weights(self):

        with torch.no_grad():

            for line in range(self.out_features):

                self.weight[line, self.flipped_sign_only]=0.0

                for group in self.rotation_groups:
                    for item in group[1:]:

                        signe=sign(item/group[0])
                        self.weight[line, int(abs(item))]= signe*self.weight[line, int(abs(group[0])) ]

    def enfore_requires_grad(self, rotation_table):

        #All biases require gradients
        self.bias.requires_grad=True

        #List of index that are not rotated
        self.no_rotation=[]
        #List of indexes that just require a flipped sign
        self.flipped_sign_only=[]
        #Rotation groups
        self.rotation_groups=[]

        #For all input indexes 
        for base_index in range(self.in_features):
            
            #If no rotation, store
            if (rotation_table[base_index]==base_index):
                self.no_rotation.append(base_index)
            #If flipped sign, store

            elif (rotation_table[base_index]==-base_index):
                self.flipped_sign_only.append(base_index)
            #Else, it's a rotation

            else:
                #We look in the groups if the index is already store       
                found=False
                for group in self.rotation_groups:
                    if (base_index in group):
                        current_group=group
                        found=True 

                #If not, we create a new group
                if (not(found)):
                    self.rotation_groups.append([base_index])
                    current_group=self.rotation_groups[-1]

                #For this index, we look for the corresponding entries in the rotation table and add them to the group
                for i, rotated_index in enumerate(rotation_table):
                    if(abs(rotated_index)==base_index)and(not(i in current_group)):
                        current_group.append(i)

        #We merge groups together
        for group in self.rotation_groups:
            for item in group:
                for group2 in self.rotation_groups:
                    if(group != group2):
                        for item2 in group2:
                            if (abs(item)==abs(item2)):
                                group.extend(group2)
                                self.rotation_groups.remove(group2)

        #We remove double entries in all groups
        for i,group in enumerate(self.rotation_groups):
            self.rotation_groups[i]=list(set(group))

        #We add signs


       # print(self.no_rotation)
       # print(self.flipped_sign_only)
        print("")
        print("rotation group :",self.rotation_groups)
        print("")

        self.first_elements_rotation_groups=[]
        self.other_elements_rotation_groups=[]

        for group in self.rotation_groups:
            self.first_elements_rotation_groups.append(abs(group[0]))
            self.other_elements_rotation_groups.extend([abs(item) for item in group[1:]])

        #print(self.first_elements_rotation_groups)
        #print(self.other_elements_rotation_groups)

        with torch.no_grad():
            for line in range(self.out_features):
                self.weight[line, self.no_rotation].requires_grad_(True)
                self.weight[line, self.flipped_sign_only].requires_grad_(False)
                self.weight[line, self.first_elements_rotation_groups].requires_grad_(True)
                self.weight[line, self.other_elements_rotation_groups].requires_grad_(False)
            self.bias.requires_grad_(True)
reflexion()


