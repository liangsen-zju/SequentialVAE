import torch
import torch.nn as nn

# https://github.com/Jarvisgivemeasuit/DPANet-Pytorch/blob/master/utils/utils.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, eps=1e-7, reducation='mean'):
        super().__init__()
        self.alpha = Variable(torch.tensor(alpha))
        self.gamma = gamma
        self.eps = eps
        self.reducation = reducation

    def forward(self, pred, target):
        N = pred.shape[0]
        C = pred.shape[1]
        num_pixels = pred.shape[2] * pred.shape[3]

        target_index = target.view(target.shape[0], target.shape[1], target.shape[2], 1)
        class_mask = torch.zeros([N, pred.shape[2], pred.shape[3], C]).cuda()
        class_mask = class_mask.scatter_(3, target_index, 1.)
        class_mask = class_mask.transpose(1, 3)
        class_mask = class_mask.view(pred.shape)

        logsoft_pred = F.log_softmax(pred, dim=1)
        soft_pred = F.softmax(pred, dim=1)

        loss = -self.alpha * ((1 - soft_pred)) ** self.gamma * logsoft_pred
        loss = loss * class_mask
        loss = loss.sum(1)

        if self.reducation == 'mean':
            return loss.sum() / (class_mask.sum() + self.eps)
        else:
            return loss.sum()