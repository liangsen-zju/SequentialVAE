import torch
import torch.nn as nn

from .utils import weighted_loss

@weighted_loss
def pfld_loss(pred, target, euler_pt, euler_gt, attribute_gt):
    """PFLD loss.
    Args:
        pred (torch.Tensor): The prediction landmark. (nbatch, 68*2)
        
    
    """