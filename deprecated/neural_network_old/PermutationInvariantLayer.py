import torch.nn as nn
import math
import torch

def sign(x):
    return int(math.copysign(1,x))

class PermutationInvariantLinear(nn.Module):

    def __init__(self, in_features, out_features, permutation_table):
        super(PermutationInvariantLinear, self).__init__()

        #in_features: dimension of the input of the NN (1D)
        self.in_features=in_features

        #out_fetures: dimension of the output of the NN (1D)
        self.out_features=out_features

        print([int(k.item()) for k in permutation_table])

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
        
        # Compute the output using the weights and biases
        output = torch.matmul(x, self.weight.t()) + self.bias

        return output

    def enforce_symmetric_weights(self):

        with torch.no_grad():
            for i in range(self.in_features):
                index=self.permutation_table[i]
                if (index==self.code_zero):
                    self.weight[:,i]=0.0
                else:
                    self.weight[:,i]=self.sign_table[i]*self.weight[:,index]

    def enforce_symmetric_grad(self):

        for i in range(self.in_features):
            index=self.permutation_table[i]
            if (index==self.code_zero):
                self.weight.grad[:,i]=0.0
            else:
                self.weight.grad[:,i]=self.sign_table[i]*self.weight.grad[:,index]

    def enfore_requires_grad(self):
        done=False
        niter=0
        while (not done):
            niter+=1
            done=True
            for k in range(self.in_features):

                target=self.permutation_table[self.permutation_table[k]]

                sign_target=self.sign_table[self.permutation_table[k]]
                                            
                if (self.permutation_table[k]!=target):
                    self.permutation_table[k]=target
                    self.sign_table[k]*=sign_target
                    done=False

        for k in range(self.in_features):
            if (k==self.permutation_table[k] and self.sign_table[k]<0):
                for k2, item in enumerate(self.permutation_table):
                    if (item==k):
                        self.permutation_table[k2] = self.code_zero

        print([k for k in self.permutation_table])

        # self.bias.requires_grad_(True)

        # with torch.no_grad():
        #     for k in range(self.in_features):
        #             if (k in self.permutation_table):
        #                 self.weight[:, k].requires_grad=True
        #                 print(self.weight[:, k])
        #             else:
        #                 self.weight[:, k].requires_grad=False
        #                 print(self.weight[:, k])
