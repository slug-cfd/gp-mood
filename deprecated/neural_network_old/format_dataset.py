from NN import *
from utils import *
from symmetries import *
#Takes the data in the data/ generate NN-ready files 

max_N0_per_problem=3000
ratio=2 #N1/N0 ratio
rotation = False
ncfl=4

def format_dataset():
    
    file_names=[]
    paths = []
    N0_per_problem=dict()
    N0_per_CFL=dict()

    #First we count the data
    for file_name in os.listdir('data/'):
            if ('pt' in file_name):
                paths.append('data/'+file_name)
                file_names.append(file_name)

    #and we compute N0, the number of entry with radius 0
    N0=0
    N1=0

    for path in paths:

        data = torch.load(path)
        #N is the size of the file contained in path
        N=data.shape[0]
        #as the files are already sorted by radius 0 or 1, we now to which number N0, N1 to add N
      
        if (path[-10]=='0'):

            #if its R=0 data, I limit the amount of input
            N=min(int(max_N0_per_problem/ncfl), N)
            N0+=N

            problem=path[8:16]
            CFL=path[20:23]
            if (problem in N0_per_problem):
                N0_per_problem[problem]+=N
            else:
                N0_per_problem[problem]=N
            if (CFL in N0_per_CFL):
                N0_per_CFL[CFL]+=N
            else:
                N0_per_CFL[CFL]=N

        else:
            N1+=N

    #Count is over, some stats
    print(colors.green+"Data repartition per problem"+colors.ENDC)
    for key in N0_per_problem:
        print(key, format(N0_per_problem[key]*100/N0), "%",N0_per_problem[key])

    print(colors.green+"Data repartition per CFL"+colors.ENDC)
    for key in N0_per_CFL:
        print(key, format(N0_per_CFL[key]*100/N0), "%")
        
    #Allocate the torch arrays
    print("Total is ", N0,"entries with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))

    k0=0
    k1=0

    #Browse the file again and gather the data
    #We shuffle to have data from all time steps, not just from the beginning when Nreal!=N
    for path in paths:
        with open(path, 'r') as file:
                
                data = torch.load(path)

                #if its R=0 data, I limit the amount of input

                if (path[-10]=='0'):   
                    N=min(int(max_N0_per_problem/ncfl),data.shape[0])
                else:
                    N=data.shape[0]

                Nreal=data.shape[0]

                indexes = random.sample(range(0, Nreal), N)

                if (path[-10]=='0'):   
                    input0[k0:k0+N,:]=data[indexes,:]
                    k0+=N

                elif (path[-10]=='1'):
                    input1[k1:k1+N,:]=data[indexes,:]
                    k1+=N
                else:
                    print("problem")
                    sys.exit()
    
    print("should be 0",k1-N1,k0-N0)
    
    #We re-shuffle to mix problems
    indexes = random.sample(range(0, N0), N0)
    input0[:,:]=input0[indexes,:]
    indexes = random.sample(range(0, N1), N1)
    input1[:,:]=input1[indexes,:]

    # pick the training size as 1+ratio times the amount of N0 entries          
    data_size=int(N0*(1+ratio))

    print("Dataset has", data_size, "entries with", N0, "R=0 inputs")

    inputs=torch.ones((data_size,L))*-666
    labels=torch.ones((data_size,2))*-666

    #Get all the N0, R=0 data and N0*(1+ratio)R=1 data
    for i in range(N0):
        inputs[i,:]=input0[i,:]
        labels[i,0]=1.0
        labels[i,1]=0.0
    print('i=',i,"N0=",N0, 'data size',data_size)
    i0=i
    for i in range(i0+1, data_size):
        inputs[i,:]=input1[i-i0,:]
        labels[i,0]=0.0
        labels[i,1]=1.0
    
    for i in range(inputs.shape[0]):  # iterate over rows
        for j in range(inputs.shape[1]):  # iterate over columns
            # access the current element
            element = inputs[i, j].item()
            if (element==-666):
                print("problem")
                sys.exit()
            # do something with the current element
    
    print('should be 0',i-(data_size-1))

    torch.save({'inputs': inputs, 'labels': labels}, 'dataset.pt')
    print('done')
    #rotated datset

    if (rotation):
        print("Rotated dataset  has", 4*data_size, "entries with", 4*N0, "R=0 inputs")
        inputs_rot=torch.zeros((data_size*4,L))
        labels_rot=torch.zeros((data_size*4,2))

        # for i in range(data_size):
        #     data_rotated=rotate(inputs[i,:])
        #     for nrot in range(4):
        #         inputs_rot[4*i + nrot,:]=data_rotated[nrot]
        #         labels_rot[4*i + nrot,:]=labels[i,:]

        p_table=compute_permutation_table_90_rot()
        data_rotated=[torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666]

        for i in range(data_size):

            data_rotated[0]=inputs[i,:]

            for nrot in range(1,4):
                data_rotated[nrot]=rotate_90(data_rotated[nrot-1], p_table)
            for nrot in range(4):
                inputs_rot[4*i + nrot,:]=data_rotated[nrot]
                labels_rot[4*i + nrot,:]=labels[i,:]
            
            print(i, data_size)

        torch.save({'inputs': inputs_rot, 'labels': labels_rot}, 'dataset_rot.pt')

format_dataset()