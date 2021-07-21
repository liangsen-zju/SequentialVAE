import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as dist
from pathlib import Path
from collections import OrderedDict

from src.models import init_net
from src.models.base import Base
from src.optims import get_optimizer, get_scheduler

from common.image import tensor2im
from common.animation import save_animation
from common.debug import debug


# disentangle model
class ConditionalVAEModel(Base):

    def __init__(self, opt, logger):
        Base.__init__(self, opt, logger)

        self.opt = opt
        self.logger = logger

        self.names_model = ["G"]              # G

        # define networks
        self.netG = define_netG(opt)
        self.netG.to(f"cuda:{opt.GPUS[0]}")
        self.netG = torch.nn.DataParallel(self.netG, opt.GPUS)  # multi-GPUs
        init_net(self.netG, init_type='kaiming', gpu_ids=opt.GPUS)
            

        if self.isTrain:
            self.names_loss = opt.LOSS.names

            # define optimizers
            self.optimizer_G = get_optimizer(self.netG, opt)
            self.optimizers.append(self.optimizer_G)

            self.scheduler_G = get_scheduler(self.optimizer_G, opt)
            self.schedulers.append(self.scheduler_G)
    

    def forward(self):
        pass


    def set_input(self, batch_input):

        self.data = batch_input[0]      # (B, 1, N, 28)
        self.anno = batch_input[1]      # (B, )               

        self.data = self.data.float().to(self.device)
        self.data = self.data.squeeze(1)                     # (B, N, 28)
        self.B, self.N = self.data.size(0), self.data.size(1)

        self.anno = self.anno.float().to(self.device)
        self.anno = self.anno.reshape(self.B, 1, 1)
        self.anno = self.anno.repeat(1, self.N,  1)               # (B, N, 1)

        # print(f"data shape = {self.data.shape}, anno shape = {self.anno.shape}")


    def update_parameters(self, epoch):
        # update beta
        n_period = self.opt.TRAIN.epoch_end / self.opt.TRAIN.n_cycle 
        n_curr = epoch % n_period 


        if n_curr < max(10, 0.1 * n_period):
            step = 0.0001
            beta_update = 0.0001 * n_curr

        elif n_curr < 0.4 * n_period:
            n_start = max(10, 0.1 * n_period)
            n_end   = 0.4 * n_period
            step = (self.opt.TRAIN.beta_stop - 0.0001*n_start ) / (n_end - n_start) 
            beta_update =  0.0001*n_start + step * (n_curr - n_start)

        elif n_curr < 0.6 * n_period:
            beta_update = self.opt.TRAIN.beta_stop
        
        else:
            step = (self.opt.TRAIN.beta_stop - self.opt.TRAIN.beta_start) / (0.4 * n_period )
            beta_update = step * (n_period - n_curr)
        

        self.opt.LOSS.lambda_kld = beta_update

        self.logger.info(f"\t[UPDATE] lambda_kld = {self.opt.LOSS.lambda_kld}")



    def training(self, batch_input):
        self.train()     # set train model
        self.set_input(batch_input)
        
        # forward
        self.pred = self.netG(self.data, self.anno)                                 # (B, N, 28)

        # loss
        loss, self.loss_dict = self.netG.module.eval_losses()

        # backward
        self.optimizer_G.zero_grad()
        loss.backward()
        
        if self.opt.TRAIN.grad_clip_norm:  # operate grad
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.opt.TRAIN.max_grad_clip)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.TRAIN.max_grad_norm)
        
        self.optimizer_G.step()

        return loss
        
    def validation(self, batch_input):
        self.eval()    # set eval model
        with torch.no_grad():
            pass


    def testing(self, batch_input=None, cur_epoch=0, cur_batch=0):
        self.eval()    # set eval model

        # generate cond
        cond = torch.tensor([0,1,2,3,4,5,6,7,8,9]).reshape(10, 1, 1)
        cond = cond.repeat(10, 28,  1)                                 # (B, N, 1)
        cond = cond.float().to(self.device)

        with torch.no_grad():
            samples = self.netG.module.generate(cond, nsteps=28, device=self.device, nbatch=100)  # (B, N, 28)
            samples = samples.cpu().numpy().reshape(10, 10, 28, 28)
            samples = samples.transpose(0, 2, 1, 3).reshape(10*28, 10*28)
        
            path_save =  Path(self.opt.OUTPUT_DIR,f"generation-{cur_epoch:04d}-{cur_batch:04d}.png")
            samples = (samples * 255).astype(np.uint8)
            cv2.imwrite(str(path_save), samples)
         
    def get_test_visuals(self):
        """Return visualization images. train.py will display these images with 
        visdom, and save the images to a HTML"""
        visual_dict = OrderedDict()
        return visual_dict


    def get_train_visuals(self, isTrain=True):
        """Return visualization images. train.py will display these images with 
        visdom, and save the images to a HTML"""

        visual_dict = OrderedDict()
        batch_data = self.data.detach().cpu().numpy()   # (B, 28, 28)
        batch_pred = self.pred.detach().cpu().numpy()   # (B, 28, 28)


        image_data = batch_data[:100, :, :].reshape(10, 10, 28 ,28)
        image_data = image_data.transpose(0, 2, 1, 3).reshape(10*28, 10*28, 1)
        visual_dict[f'GT']  = (image_data * 255).astype(np.uint8)


        image_pred = batch_pred[:100, :, :].reshape(10, 10, 28 ,28)
        image_pred = image_pred.transpose(0, 2, 1, 3).reshape(10*28, 10*28, 1)
        visual_dict[f'PT']  = (image_pred * 255).astype(np.uint8)


        # samples
        with torch.no_grad():
            # cond = self.anno[:100, :, :]
            cond = torch.tensor([0,1,2,3,4,5,6,7,8,9]).reshape(10, 1, 1)
            cond = cond.repeat(10, 28,  1)                                 # (B, N, 1)
            cond = cond.float().to(self.device)

            samples = self.netG.module.generate(cond, nsteps=28, device=self.device, nbatch=100)  # (B, N, 28)
            samples = samples.cpu().numpy().reshape(10, 10, 28, 28)
            samples = samples.transpose(0, 2, 1, 3).reshape(10*28, 10*28, 1)
            visual_dict[f'samples'] = (samples * 255).astype(np.uint8)

        return visual_dict


#######################
def define_netG(opt):
    netG = None
    if opt.MODEL.name_netG == "ConditionalVRNN":
        from src.nn.ConditionalVRNN import ConditionalVRNN
        netG = ConditionalVRNN(opt)
        
    elif opt.MODEL.name_netG == "VRNN":
        from src.nn.VRNN import VRNN
        netG = VRNN(opt)
    
    else:
        raise NameError(f"The model of opt.MODEL.name_netAE(={opt.MODEL.name_netG}) not existing!")


    return netG