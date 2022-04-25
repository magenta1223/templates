# preprocessing 
import time
import os
from glob import glob

# Pytorch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# my apps
from hparams import hparams
from funcs import *
from dataset import *
from models import *
from utils import *

# argument parser
import argparse

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('--window', type=int, default=8)

os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID" 
# Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3" 

args = parser.parse_args()

def main(args):


    
    hparams.window = args.window

    print("\nArguments")
    print(f"\nwindow : {hparams.window}")



    total_files = sorted(glob('./data_v2/weekly_train/*.npy') )[481:]

    num_train_files = hparams.num_train_files
    num_valid_files = hparams.num_valid_files



    train_dataset = MyDataset(hparams)
    valid_dataset = MyDataset(hparams)


    train_loader = DataLoader(
        train_dataset,
        batch_size= hparams.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers= hparams.num_workers)

    val_loader = DataLoader(
        valid_dataset,
        batch_size= hparams.batch_size,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        drop_last=False,
        num_workers= hparams.num_workers)

    
    m = MyModel(hparams)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    #m = nn.DataParallel(m)

    m.to(device)
    

    # loss : mse, mae, logcosh, msssim mae_over_f1, mse_ovef_f1
    # multisteplr 
    fitter = Fitter(model = m, config = hparams)
    fitter.fit(train_loader, val_loader, False)


if __name__ == "__main__":
    main(args)
