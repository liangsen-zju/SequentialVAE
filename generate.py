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
    args = parese_args()

    path_config = Path(args.opt)
    
    postfix = opt.TRAIN.resume_suffix if opt.TRAIN.resume else None           # if resume, resume to resume_suffix folder
    logger = Logger(opt.OUTPUT_DIR, path_config=path_config, postfix=postfix) 
    logger.info(pprint.pformat(opt))
    opt.OUTPUT_DIR = logger.path_output

    # model 
    model = create_model(opt, logger)
    model.setup(opt)
    model.testing()                   # forward and backward