# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict

class MultiLossManager(nn.Module):
    """Managment Multiple-Loss
    """

    def __init__(self, opt, autoWeight=False):
        super(MultiLossManager, self).__init__()

        self.opt = opt
        self.autoWeight = autoWeight

        self.weight = {}
        self.weigth_name = []

        for iname in opt.LOSS.names:
            ifull = f"lambda_{iname}"
            if ifull in opt.LOSS.keys() and opt.LOSS[ifull] > 0:
                self.weight[ "loss_"+iname ] = opt.LOSS[ifull]
                self.weigth_name.append("loss_" + iname )
        n_weight = len(self.weight)

        print(f"WEIGHT of LOSS (n={n_weight}) = {self.weight}")

        if autoWeight:
            self.weight = torch.ones(n_weight, requires_grad=True)
            self.weight = torch.nn.Parameter(self.weight)


    def forward(self, dict_loss):
        assert isinstance(dict_loss, OrderedDict), "dict_loss should be a OrderedDict"

        loss_sum = 0
        i = 0
        for name in dict_loss.keys():
            if name not in self.weigth_name: continue

            if self.autoWeight:
                loss_sum += 0.5 / (self.weight[i] ** 2) * dict_loss[name] + torch.log(1 + self.weight[i] ** 2)
            else:
                loss_sum += (self.weight[name] * dict_loss[name])

            i += 1
        
        # print(f"update WEIGHT of LOSS {self.weight}")

        return loss_sum



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)


    def forward(self, dict_loss):

        assert isinstance(dict_loss, OrderedDict), "dict_loss should be a OrderedDict"

        loss_sum = 0
        for i, name in enumerate(dict_loss.keys()):

            loss_sum += 0.5 / (self.params[i] ** 2) * dict_loss[name] + torch.log(1 + self.params[i] ** 2)
        
        return loss_sum


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())