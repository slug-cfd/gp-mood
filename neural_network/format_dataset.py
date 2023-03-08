from NN import *
import os
import random

#Import all the training data
path = '../gp-mood-code/'
paths = []
for file_name in os.listdir(path):
    if "torch.pt" in file_name:
        paths.append(path+file_name)

N0=0
N1=0

for path in paths:
    data = torch.load(path)
    N=data.shape[0]
    print(N, data.shape)

    if (path[-10]=='0'):
        N0+=N
    else:
        N1+=N

print("total R=0",N0,"R=1",N1)
input0=torch.zeros((N0,57))
input1=torch.zeros((N1,57))
print(N0, N1)

k0=0
k1=0

for path in paths:
    with open(path, 'r') as file:
                
            data = torch.load(path)
            N=data.shape[0]
            if (path[-10]=='0'):
                print("0",path)
                input0[k0:k0+N,:]=data[:,:]
                k0+=N
            else:
                print("1",path)
                input1[k1:k1+N,:]=data[:,:]
                k1+=N
                

#Define the training set
training_size=2*N0
inputs=torch.zeros((training_size,57))
labels=torch.zeros((training_size,2))

for i in range(training_size):
     
     if (i%2==0):
          k=int(i/2)
          inputs[i,:]=input0[k,:]
          labels[i,0]=1.0
          labels[i,1]=0.0
     else:
          k=random.randint(1, N1-1)
          #print(k, i,input1.shape)
          inputs[i,:]=input1[k,:]
          labels[i,0]=0.0
          labels[i,1]=1.0
     print(i, training_size)

torch.save(inputs, 'inputs.pt')
torch.save(labels, 'labels.pt')
