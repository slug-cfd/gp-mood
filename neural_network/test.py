









#NN(x0,x1)= NN(x1,x0)
#w00=w01
#w11=w10



class MyLayer(nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(2, 2))
        self.bias = nn.Parameter(torch.randn(2))
        
        self.weight[0, 0].requires_grad = True
        self.weight[0, 1].requires_grad = False
        self.weight[1, 0].requires_grad = False
        self.weight[1, 1].requires_grad = True

    def forward(self, x):
        self.weight[0, 1] = self.weight[0,0]
        self.weight[1, 0] = self.weight[1,1]

        # Compute the output using the modified weights and biases
        output = torch.matmul(x, self.weight.t()) + self.bias
        return output