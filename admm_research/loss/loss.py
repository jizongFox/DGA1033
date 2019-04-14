import torch
import torch.nn as nn
import torch.nn.functional as F
from admm_research.utils import simplex


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)

class PartialCrossEntropyLoss2d(nn.Module):

    def __init__(self,reduce=True, size_average=True):
        super(PartialCrossEntropyLoss2d, self).__init__()
        weight = torch.Tensor([0,1])
        self.loss = nn.NLLLoss(weight=weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class MSE_2D(nn.Module):
    def __init__(self):
        super(MSE_2D, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        prob = F.softmax(input, dim=1)[:, 1].squeeze()
        target = target.squeeze()
        assert prob.shape == target.shape
        return self.loss(prob, target.float())
class Entropy(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        the definition of Entropy is - \sum p(xi) log (p(xi))
        '''

    def forward(self, input: torch.Tensor):
        assert input.shape.__len__() >= 2
        b, _, *s = input.shape
        assert simplex(input)
        e = input * (input + 1e-16).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        return e