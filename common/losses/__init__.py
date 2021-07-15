from .wing import WingLoss
from .smooth_l1_loss import SmoothL1Loss, L1Loss
from .landmarkloss import LandmarkPolygonLoss, LandmarkFocalLoss
"""
Source: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/__init__.py

"""

__all__ = [
    "WingLoss", "AdaptiveWingLoss", "LandmarkPolygonLoss", "LandmarkFocalLoss"
]

