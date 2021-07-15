import math
import torch
from torch import nn

from .utils import weighted_loss

"""
The Wing loss:

It has been proposed in `Wing Loss for Roubst Facial Landmark Localisation with 
Convolutional Neural Networks`. https://arxiv.org/abs/1711.06753

"""


@weighted_loss
def wing_loss(pred, target, w=5, epsilon=0.5):
    """
    Source: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py

    """

    diff = torch.abs(pred - target)
    C = w - w * math.log(1 + w / epsilon)
    loss = torch.where(diff < w,  w * torch.log(1 + diff / epsilon), diff - C)

    return loss



class WingLoss(nn.Module):

    def __init__(self, w=10, epsilon=0.5, reduction='mean', loss_weight=1.0):
        super(WingLoss,self).__init__()

        self.w = w
        self.epsilon = epsilon
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.
        Args:
            pred (torch.Tensor, Bx68*2): The prediction
            target (torch.Tensor, Bx68*2): The GT of prediction
            weight (torch.Tensor, optional): The weight of each loss, default None
            avg_factor(int, optional): The average of the reduction loss
        """

        loss = wing_loss(pred, target, w=self.w, epsilon=self.epsilon, weight=weight, avg_factor=avg_factor)
        loss = self.loss_weight * loss

        return loss







if __name__ == "__main__":
    crieation = WingLoss()

    GT = torch.zeros((2, 68, 64, 64))
    PT = torch.ones((2, 68, 64, 64))
    PT.requires_grad_(True)

    loss = crieation(PT, GT)
    loss.backward()

    print(loss)


