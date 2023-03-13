from NN import *
import os
import random
from utils import *

#Takes the data in the training/ and testing/ folders an generate NN-ready files 

def format_dataset(type, name):

    #create a folder with an experiment name
    folder='sets_'+name
    os.system('mkdir '+folder)

    #fodler where to get the data to format
    path = type+'/'

    #path where to write the NN-ready data
    label_path=folder+'/'+type+'_label.pt'
    inputs_path=folder+'/'+type+'_inputs.pt'

    print(colors.HEADER+"Reading the data inpy", path, "and storing it in", label_path, inputs_path+colors.ENDC)
    
    file_names=[]
    paths = []
    N0_per_problem=dict()
    N0_per_CFL=dict()

    #First we count the data
    #We browse all file
    for file_name in os.listdir(path):
        paths.append(path+file_name)
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
            N0+=N

            problem=path[12:19]
            CFL=path[24:27]

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
        print(key, N0_per_problem[key]*100/N0, "%")
    print(colors.green+"Data repartition per CFL"+colors.ENDC)
    for key in N0_per_CFL:
        print(key, N0_per_CFL[key]*100/N0, "%")
        
    #Allocate the torch arrays
    print(N0,"entry with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))
    print("Saving a dataset with ",N0, "R=0 entries", "and", N0*(2-1), "R=1 entries")

    k0=0
    k1=0

    #Browse the file again and gather the data
    for path in paths:
        with open(path, 'r') as file:
                    
                data = torch.load(path)
                N=data.shape[0]
                if (path[-10]=='0'):
                    input0[k0:k0+N,:]=data[:,:]
                    k0+=N
                else:
                    input1[k1:k1+N,:]=data[:,:]
                    k1+=N

    # pick the training size as two times the amount of N0 entries          
    training_size=2*N0
    inputs=torch.zeros((training_size,L))
    labels=torch.zeros((training_size,2))

    #Get all the N0 data and N0 N1 data
    for i in range(training_size):
        
        if (i%2==0):
            k=int(i/2)
            inputs[i,:]=input0[k,:]
            labels[i,0]=1.0
            labels[i,1]=0.0
        else:
            k=random.randint(0, N1-1)
            inputs[i,:]=input1[k,:]
            labels[i,0]=0.0
            labels[i,1]=1.0

    print("saving inputs/labels in ", folder, "as well as in the current directory")
    torch.save(inputs, inputs_path)
    torch.save(labels, label_path)

    torch.save(inputs, type+'_inputs.pt')
    torch.save(labels, type+'_labels.pt')

    print("saving logs ...")

    f = open(folder+"/log_"+type+".txt", "w")
    f.write("the "+type+" dataset was made with 2= "+str(2) +"and: \n")
    for name in file_names:
        f.write(name+'\n')
    f.close()

    print('end')




type=sys.argv[1]
if (len(sys.argv)<3)or((type!='training')and(type!='testing')):
    print(type)
    print("error, usage: python3 format_datset.py, training / testing, name")
    sys.exit()

name = sys.argv[2]


format_dataset(type, name)