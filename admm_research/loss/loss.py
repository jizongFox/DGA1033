import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class PartialCrossEntropyLoss2d(nn.Module):

    def __init__(self, reduce='mean', size_average=True, enforce_nCE=False):
        super(PartialCrossEntropyLoss2d, self).__init__()
        weight = torch.Tensor([0, 1])
        self.loss = nn.NLLLoss(weight=weight, reduce=reduce, size_average=size_average)
        self.loss_0 = nn.NLLLoss(reduce=reduce, size_average=size_average)
        self.enforce_nCE = enforce_nCE

    def forward(self, outputs, targets):

        return self.loss(F.log_softmax(outputs, dim=1), targets)

class negativePartialCrossEntropyLoss2d(nn.Module):

    def __init__(self, reduce='mean', size_average=True, enforce_nCE=False):
        super(negativePartialCrossEntropyLoss2d, self).__init__()
        weight = torch.Tensor([0, 1])
        self.loss = nn.NLLLoss(weight=weight, reduce=reduce, size_average=size_average)
        self.loss_0 = nn.NLLLoss(reduce=reduce, size_average=size_average)
        self.enforce_nCE = enforce_nCE

    def forward(self, outputs, targets):
        if self.enforce_nCE:
            negative_idx = (targets.sum([1, 2]) == 0).type(torch.ByteTensor)
            positive_idx = (targets.sum([1, 2]) > 0).type(torch.ByteTensor)
            n_targets, n_outputs = targets[negative_idx], outputs[negative_idx]
            p_targets, p_outputs = targets[positive_idx], outputs[positive_idx]
            p_loss = self.loss(F.log_softmax(p_outputs, dim=1), p_targets)
            if negative_idx.float().allclose(torch.Tensor([0])):
                n_loss = torch.Tensor(0)
            else:
                n_loss = self.loss_0(F.log_softmax(n_outputs, dim=1), n_targets)
            loss = p_loss * p_targets.shape[0] + n_loss * n_targets.shape[0]
            loss = loss / outputs.shape[0]
            return loss

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

