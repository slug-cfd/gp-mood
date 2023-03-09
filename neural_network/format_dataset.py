from NN import *
import os
import random
from utils import *

#Takes the data in the training/ and testing/ folders an generate NN-ready files

def format_dataset(type, ratio, name):

    folder='sets_'+name

    path = type+'/'
    label_path=folder+'/'+type+'_label.pt'
    inputs_path=folder+'/'+type+'_inputs.pt'

    os.system('mkdir '+folder)

    print(colors.HEADER+"Reading the data inpy", path, "and storing it in", label_path, inputs_path+colors.ENDC)
    
    file_names=[]
    paths = []
    for file_name in os.listdir(path):
        paths.append(path+file_name)
        file_names.append(file_name)
        print("reading ", path+file_name, " ... ")

    N0=0
    N1=0

    for path in paths:
        data = torch.load(path)
        N=data.shape[0]

        if (path[-10]=='0'):
            N0+=N
        else:
            N1+=N

    print(N0,"entry with R=0 and", N1, " with R=1", (100*N0)/(N1+N0), "%")
    input0=torch.zeros((N0,L))
    input1=torch.zeros((N1,L))
    print("Saving a dataset with ",N0, "R=0 entries", "and", N0*(ratio-1), "R=1 entries")

    k0=0
    k1=0

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
                    
    training_size=ratio*N0
    inputs=torch.zeros((training_size,L))
    labels=torch.zeros((training_size,2))

    for i in range(training_size):
        
        if (i%ratio==0):
            k=int(i/ratio)
            inputs[i,:]=input0[k,:]
            labels[i,0]=1.0
            labels[i,1]=0.0
        else:
            k=random.randint(1, N1-1)
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
    f.write("the "+type+" dataset was made with ratio= "+str(ratio) +"and: \n")
    for name in file_names:
        f.write(name+'\n')
    f.close()

    print('end')




type=sys.argv[1]
if (len(sys.argv)<4)or((type!='training')and(type!='testing')):
    print(type)
    print("error, usage: python3 format_datset.py, training / testing, ratio, name")
    sys.exit()

ratio = int(sys.argv[2])
name = sys.argv[3]


format_dataset(type, ratio, name)