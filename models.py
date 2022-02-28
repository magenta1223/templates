from modules import *
import torch
import torch.nn as nn
from utils import *

# for mixed precision
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


class MyModel(nn.Module):
    def __init__(self, hparams):
        super(MyModel, self).__init__()

        # -------------------------------MODEL------------------------------- #    
    
        
        # -------------------------------Optimizer--------------------------- #
        self.optimizer  
        self.loss_fn 
        self.accumulative_step

        # -------------------------------Scheduler--------------------------- #
        self.scheduler = hparams.SchedulerClass(self.optimizer, **hparams.scheduler_params) # scheduler


        # -------------------------------Metric------------------------------ #
        self.loss_meters = {'loss' : AverageMeter()}

        # -------------------------------Mixed Precision--------------------- #
        self.scaler = GradScaler()



    def forward(self, x):
        
        return 
        

    def optimize(self, batch, iter):
        # -------------------------------TRAIN------------------------------- #

        inputs, targets = batch

        inputs = inputs.cuda()
        targets = targets.cuda()


        with autocast():
            out = self(inputs) 
            loss = self.loss_fn(out, targets ) / self.accumulative_step


        # -------------------------------DO NOT EDIT------------------------- #
        self.scaler.scale(loss).backward() # mixed precision


        # gradient accumulation 
        if not (iter + 1) % self.accumulative_step: 
            #self.optimizer.step() 
            self.scaler.step(self.optimizer) # mixed precision       
            self.scaler.update()   
            self.optimizer.zero_grad()

        self.loss_log(loss, batch[0].size(0) )


    def validate(self, batch):
        # -------------------------------VALIDATE--------------------------- #
        batch = batch.cuda()
        inputs, targets = batch
        out = self(inputs) 
        loss = self.loss_fn(out, targets)   
        self.loss_log(loss, batch[0].size(0) )


        # -------------------------------DO NOT EDIT------------------------- #
    def loss_log(self, loss, batch_size, key = 'loss'):
        self.loss_meters[key].update(loss.item() * self.accumulative_step, batch_size )

    def loss_dict(self):
        return { k : v.avg for k, v in self.loss_meters.items() }
    
    def meter_initialize(self):
        for k, v in self.loss_meters.items():
            v.reset()
