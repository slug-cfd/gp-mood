from NN import *
from utils import *

#Takes the data in the data/ generate NN-ready files 

max_N0_per_problem=11500
ratio=1 #N1/N0 ratio

def format_dataset():
    
    file_names=[]
    paths = []
    N0_per_problem=dict()
    N0_per_CFL=dict()

    #First we count the data
    for file_name in os.listdir('data/'):
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
            N=min(int(max_N0_per_problem/3), N)
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
        print(key, N0_per_problem[key]*100/N0, "%",N0_per_problem[key])

    print(colors.green+"Data repartition per CFL"+colors.ENDC)
    for key in N0_per_CFL:
        print(key, N0_per_CFL[key]*100/N0, "%")
        
    #Allocate the torch arrays
    print(N0,"entries with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))

    k0=0
    k1=0

    #Browse the file again and gather the data
    #We shuffle to have data from all time steps, not just from the beginning when Nreal!=N
    for path in paths:
        with open(path, 'r') as file:
                    
                data = torch.load(path)

                N=data.shape[0]
                Nreal=data.shape[0]

                if (path[-10]=='0'):   

                    #if its R=0 data, I limit the amount of input
                    N=min(int(max_N0_per_problem/3),N)
                    indexes = random.sample(range(0, Nreal), N)

                    input0[k0:k0+N,:]=data[indexes,:]
                    k0+=N

                else:
                    indexes = random.sample(range(0, Nreal), N)
                    input1[k1:k1+N,:]=data[indexes,:]
                    k1+=N
    
    print("k1",k1, "N1",N1, "k0",k0, "N0",N0)
    
    #We re-shuffle to mix problems
    indexes = random.sample(range(0, N0), N0)
    input0[:,:]=input0[indexes,:]
    indexes = random.sample(range(0, N1), N1)
    input1[:,:]=input1[indexes,:]


    # pick the training size as two times the amount of N0 entries          
    data_size=int(N0*(1+ratio))

    inputs=torch.zeros((data_size,L))
    labels=torch.zeros((data_size,2))

    #Get all the N0 data and N0 N1 data
    for i in range(int(data_size/(ratio+1))):
        inputs[i,:]=input0[i,:]
        labels[i,0]=1.0
        labels[i,1]=0.0
    print('i=',i, 'data size',data_size)
    for i in range(int(data_size/(ratio+1)), data_size):
        inputs[i,:]=input1[i,:]
        labels[i,0]=0.0
        labels[i,1]=1.0
    print('i=',i, 'data size',data_size)

    torch.save({'inputs': inputs, 'labels': labels}, 'dataset.pt')

format_dataset()