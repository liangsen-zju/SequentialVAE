import torch
import torch.nn as nn
import numpy as np

# from shapely.geometry import Polygon


class LandmarkPolygonLoss(nn.Module):
    def __init__(self, opt):
        super(LandmarkPolygonLoss, self).__init__()
        
        self.opt = opt
        self.lmdk_indx = {
            "chin": list(range(0,17)),
            "eyebrow_lt": [17, 18, 19, 20, 21],
            "eyebrow_rt": [22, 23, 24, 25, 26],
            "nose": [27, 31, 32, 33, 34, 35],
            "eye_lt": [36, 37, 38, 39, 40, 41],
            "eye_rt": [42, 43, 44, 45, 46, 47],
            "mouth_out": list(range(48, 60)),
            "mouth_inn": list(range(60, 68)),
        }

    def forward(self, pred, target):
        # predit, (nbatch, 68*2)
        # target, (nbatch, 68*2)

        #print(f"pred shape = {predit.shape}, target shape = {target.shape}")
        pred = pred.reshape(-1, 68, 2) * 128
        target = target.reshape(-1, 68, 2) * 128

        loss_chin = self.area_diff(pred, target, "chin")
        loss_nose = self.area_diff(pred, target, "nose")
        loss_eyebrow = self.area_diff(pred, target, "eyebrow_lt") \
                     + self.area_diff(pred, target, "eyebrow_rt")
        loss_eye = self.area_diff(pred, target, "eye_lt") \
                 + self.area_diff(pred, target, "eye_rt")
        loss_mouth = self.area_diff(pred, target, "mouth_out") \
                   + self.area_diff(pred, target, "mouth_inn")

        loss = loss_chin + loss_nose + loss_eyebrow + loss_eye # + loss_mouth
        loss = torch.mean(loss)
        
        return loss



    def area_diff(self, pred, target, part):
        indx = self.lmdk_indx[part]

        area_pt = self.PolygonArea(pred, indx)
        area_gt = self.PolygonArea(target, indx)

        diff = torch.abs(area_pt - area_gt)
        return diff


    def PolygonArea(self, corners, indx):
        """PolygonArea Function.
        Args:
            corners (torch.Tensor): the keypoints of corners, (nbatch, 68, 2)
            indx (list, int): the polygon index from the keypoints
        Return:
            area (torch.Tensor): (nbatch, )
            
        """
        area, n = 0, len(indx)
        for i in range(n):
            j = (i + 1) % n
            area += corners[:, i, 0] * corners[:, j, 1]
            area -= corners[:, j, 0] * corners[:, i, 1]

        area = torch.abs(area) / 2.0
        return area






class LandmarkFocalLoss(nn.Module):
    def __init__(self, opt):
        super(LandmarkFocalLoss, self).__init__()
        self.opt = opt


    def forward(self, predit, target):

        # predit, (nbatch, 68*2)
        # target, (nbatch, 68*2)

        #print(f"pred shape = {predit.shape}, target shape = {target.shape}")
        predit = predit.reshape(-1, 68, 2)
        target = target.reshape(-1, 68, 2)

        x = torch.abs(predit - target)                              # (nbatch, 68, 2)
        x = torch.sqrt(x[:,:,0] * x[:,:,0] + x[:,:,1] * x[:,:,1])   # (nbatch, 68)

        #print(f"xsqrt = {x[0, :]}")

        x = torch.torch.pow(1 + 1000*x, x) - 1                               # (nbatch, 68)
        x = torch.clamp_max_(x, 10)

        #print(f"xpow = {x[0, :]}")
        weight = torch.ones((68,), dtype=x.dtype, device=x.device)
        weight[48:68] = 10
        

        x = torch.mean(x, dim=0)                                              # (68, )
        x = torch.mean(x * weight)                                            # (1, )     

        return x



