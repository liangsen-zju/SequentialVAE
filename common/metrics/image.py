import torch
import numpy as np
from sklearn.metrics import f1_score as fs
# https://github.com/Jarvisgivemeasuit/DPANet-Pytorch/blob/master/utils/utils.py

class PixelAccuracy:
    def __init__(self, ignore_index=-1, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_index = ignore_index
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_index
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        self.num_correct += ((pred.long() == target.long()) * ignore_mask).sum().item()
        self.num_instance += ignore_mask.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class MeanIoU:
    def __init__(self, num_classes, eps=1e-7):
        if num_classes == 1:
            self.num_classes = num_classes + 1
        else:
            self.num_classes = num_classes
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
        self.eps = eps

    def update(self, pred, target):
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        for cur_cls in range(self.num_classes):
            pred_mask = (pred == cur_cls).byte()
            target_mask = (target == cur_cls).byte()

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()

            self.num_intersection[cur_cls] += intersection.item()
            self.num_union[cur_cls] += union.item()

    def get(self, ignore_background=False):
        if ignore_background:
            iou_list = (self.num_intersection[:-1] / (self.num_union[:-1] + self.eps))
            return iou_list.mean(), iou_list.max(), np.where(iou_list == iou_list.max()), iou_list.min(), np.where(iou_list == iou_list.min())
        else:
            iou_list = (self.num_intersection / (self.num_union + self.eps))
            return iou_list.mean(), iou_list.max(), np.where(iou_list == iou_list.max()), iou_list.min(), np.where(iou_list == iou_list.min())

    def reset(self):
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
    
    def get_all(self):
        return (self.num_intersection / (self.num_union + self.eps))

class Kappa:
    def __init__(self, num_classes):
        self.pre_vec = np.zeros(num_classes)
        self.cor_vec = np.zeros(num_classes)
        self.tar_vec = np.zeros(num_classes)
        self.num = num_classes

    def update(self, output, target):
        pre_array = torch.argmax(output, dim=1)

        for i in range(self.num):
            pre_mask = (pre_array == i).byte()
            tar_mask = (target == i).byte()
            self.cor_vec[i] = (pre_mask & tar_mask).sum().item()
            self.pre_vec[i] = pre_mask.sum().item()
            self.tar_vec[i] = tar_mask.sum().item()

    def get(self):
        assert len(self.pre_vec) == len(self.tar_vec) == len(self.pre_vec)
        tmp = 0.0
        for i in range(len(self.tar_vec)):
            tmp += self.pre_vec[i] * self.tar_vec[i]
        pe = tmp / (sum(self.tar_vec) ** 2 + 1e-8)
        p0 = sum(self.cor_vec) / (sum(self.tar_vec) + 1e-8)
        cohens_coefficient = (p0 - pe) / (1 - pe)
        return cohens_coefficient

    def reset(self):
        self.pre_vec = np.zeros(self.num)
        self.cor_vec = np.zeros(self.num)
        self.tar_vec = np.zeros(self.num)


class F1:
    def __init__(self):
        self.score = 0
        self.num = 0
        self.all = np.zeros(16)
        self.map = None
        self.map_tar = None

    def update(self, output, target):
        output = torch.argmax(output, dim=1).reshape(-1).cpu() 
        target = target.reshape(-1).cpu()
        # output = list(output)
        # target = list(target)
        self.score += fs(output, target, average='macro')
        self.num += 1
        if self.map == None:
            self.map = output
        else:
            self.map = torch.cat([self.map, output])
        if self.map_tar == None:
            self.map_tar = target
        else:
            self.map_tar = torch.cat([self.map_tar, target])

    def get(self):
        return self.score / self.num

    def reset(self):
        self.score = 0
        self.num = 0

    def get_all(self):
        return fs(self.map, self.map_tar, average=None)