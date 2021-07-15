import torch
import torch.nn as nn

from .utils import weighted_loss

@weighted_loss
def l1_loss(pred, target):
    """ L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target(torch.Tensor): The ground truth of the prediction
    
    Returns:
        torch.Tensor: loss
    """

    assert pred.size() == target.size() and target.numel() > 0

    loss = torch.abs(pred - target)

    return loss

@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 Loss.
    Args:
        pred (torch.Tensor): The prediction
        target (torch.Tensor): The GT of the prediction
    
    Returns:
        torch.Tensor: loss
    """

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,  diff - 0.5 * beta)

    return loss



class L1Loss(nn.Module):
    """L1 Loss
    Args:
        reduction (str, optional): The method to reduce the loss, Options are `none|mean|sum`
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    
    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.

        Args:
            pred (troch.Tensor): The prediction
            target (torch.Tensor): The GT of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each prediction.
            avg_factor(int, optional): Average factor that is used to average the loss.
        """

        loss = l1_loss(pred, target, weight=weight, reduction=self.reduction, avg_factor=avg_factor)
        loss = self.loss_weight * loss

        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss.
    Args:
        beta (float, optional): The threshold in the piecewise function. Default 1.0
        reduction (str, optional): The reduce loss method, options are `none|mean|sum`
        loss_weight (float, optional): The weight of loss 
    Return:
        torch.Tensor: Loss
    """

    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The GT of the prediction
            weight (torch.Tensor, optional): The weigth of loss for each prediction. Default to None
            avg_factor (int, optional): Average factor that is used to average the loss.
        """

        loss = smooth_l1_loss(pred, target, beta=self.beta, weight=weight, avg_factor=avg_factor)
        loss = self.loss_weight * loss
        return loss


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-1, times=1, eps=1e-7, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.times = times
        self.eps = eps
        if weight is None:
            self.weight = weight
        else:
            self.weight = weight.cuda()

    def forward(self, pred, target):
        mask = target != self.ignore_index
        pred = F.log_softmax(pred, dim=-1)
        loss = -pred * target
        loss = loss * mask.float()
        # print(loss, pred, target, mask)
        if self.weight is None:
            return self.times * loss.sum() / (mask.sum() + self.eps)
        else:
            return self.times * (self.weight * loss).sum() / (mask.sum() + self.eps)