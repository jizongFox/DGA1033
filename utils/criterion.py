import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None,reduce=True,size_average=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight,reduce=reduce,size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)

class MSE_2D(nn.Module):
    def __init__(self):
        super(MSE_2D,self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        prob = F.softmax(input,dim=1)[:,1].squeeze()
        target = target.squeeze()
        assert prob.shape == target.shape
        return self.loss(prob,target.float())

if __name__=="__main__":
    output = torch.rand(*(4,2,256,256))
    target = torch.randint(0,2,size=(4,256,256)).long()
    criterion = CrossEntropyLoss2d()
    loss = criterion(output,target)
    print(loss)
