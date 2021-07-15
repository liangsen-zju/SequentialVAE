import time
import sys
import pprint
import shutil
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import create_dataset
from src.models import create_model, calculate_model_params
from src.config import config as opt
from src.config import update_config

from common.vis.visualizer import Visualizer
from common.logger import Logger

torch.autograd.set_detect_anomaly(True) 

def parese_args():
    pareser = argparse.ArgumentParser(description='Audio2Expression Network')

    # general 
    pareser.add_argument("--opt", type=str, default="experiments/train-vae-GRID.yaml", help="experiment configure file name")
    args, _ = pareser.parse_known_args()
    update_config(args.opt)

    pareser.add_argument("--gpus", type=str, default="0", help="gpus")
    args = pareser.parse_args()

    return args
    


if __name__ == "__main__":
    start = time.time()
    args = parese_args()

    path_config = Path(args.opt)

    postfix = opt.TRAIN.resume_suffix if opt.TRAIN.resume else None           # if resume, resume to resume_suffix folder
    logger = Logger(opt.OUTPUT_DIR, path_config=path_config, postfix=postfix) 
    logger.info(pprint.pformat(opt))
    opt.OUTPUT_DIR = logger.path_output

    visor = Visualizer(opt, logger)

    # load data
    dataset = create_dataset(opt, phase='train')
    loader_train = DataLoader(dataset.get_train_dataset(), 
                              batch_size=opt.TRAIN.batch_size * len(opt.GPUS),
                              shuffle=opt.TRAIN.shuffle,
                              num_workers=opt.WORKERS, 
                              pin_memory=True,
                              drop_last=True)
    
    loader_valid = DataLoader(dataset.get_valid_dataset(),
                              batch_size=opt.TRAIN.batch_size * len(opt.GPUS),
                              shuffle=False,
                              num_workers=opt.WORKERS,
                              pin_memory=True,
                              drop_last=True)

    # loader_test = DataLoader(dataset.get_test_dataset(),
    #                          batch_size= opt.TEST.batch_size * len(opt.GPUS),
    #                          shuffle=False,
    #                          num_workers=opt.WORKERS,
    #                          pin_memory=True,
    #                          drop_last=False)
                              
    logger.info(f"train batch = {len(loader_train)}, valid = {len(loader_valid)}")


    # get model
    model = create_model(opt, logger)
    model.setup(opt)


    # train
    for epoch in range(opt.TRAIN.epoch_begin, opt.TRAIN.epoch_end+ 1):
        epoch_start_time = time.time()       # timer for entire epoch
        iter_data_time = time.time()         # timer for data loading per iteration
        loss_train = 0
        nbatch_train = 0

        # update learning_rate
        loss_train = loss_train / len(loader_train)
        logger.info(f"train loss = {loss_train}")
        model.update_learning_rate(loss_train)
        model.update_parameters(epoch)

        for i, ibatch_data in enumerate(loader_train):
            # logger.info(f"\n\n batch -i {i}")
            iter_start_time = time.time()    # timer for computation per iteration
            if i % opt.FREQ.batch_print == 0:
                t_data = iter_start_time - iter_data_time

            visor.reset()
            n_bacthsize = len(loader_train)

            loss = model.training(ibatch_data)                    # forward and backward
            # make_dot(loss, show_attrs=True, show_saved=True).render("attached", format="png")

            # stats
            loss_train += loss.detach()
            
            # display 
            if (i+1) % opt.FREQ.batch_display == 0 and (epoch+1) % opt.FREQ.epoch_display == 0:  # display images on visdom and save images to HTML file
                visuals = model.get_train_visuals()
                visor.display_current_results(visuals, epoch, i)

            # # generate testing result
            # if (i+1) % opt.FREQ.batch_test == 0:
            #     logger.info("Test...")
            #     for i, ibatch_data in enumerate(loader_test):
            #         model.testing(batch_input=ibatch_data, cur_epoch=epoch, cur_batch=i+1)        # forward and backward

            # print logger
            if (i+1) % opt.FREQ.batch_print == 0 and epoch % opt.FREQ.epoch_print == 0:
                losses = model.get_current_losses()   # pach to loss dict
                t_comp = (time.time() - iter_start_time) / n_bacthsize
                visor.print_current_losses(epoch, i, losses, t_comp, t_data)
                visor.plot_current_losses(epoch, float(i)/n_bacthsize, losses)
                
            # update time
            iter_data_time = time.time()

        # # validation 
        # if (epoch+1) % opt.FREQ.epoch_valid == 0:
        #     logger.info("validation...")
        #     loss_valid, nbatch_valid = 0, 0
        #     for i, ibatch_data in enumerate(loader_valid):
        #         loss_valid += model.validation(ibatch_data)                    # forward and backward        
        #     logger.info(f"validation loss = {loss_valid / len(loader_valid)}")

        # generate testing result
        if (epoch+1) % opt.FREQ.epoch_test == 0:
            logger.info("Test...")
            model.testing(cur_epoch=epoch)             # forward and backward

        # save epoch
        if (epoch + 1) % opt.FREQ.epoch_save == 0: 
            logger.info(f"saving the model at the end of epoch {epoch:04d}, iters {i:06d}")
            model.save_networks(epoch+1)


    logger.info(f"Total Time Taken: {(time.time() - start)}")



