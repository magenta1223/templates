import os
import time
from datetime import datetime
from glob import glob
import numpy as np
import torch
import torch.nn as nn 
import wandb


class Logger:
    def __init__(self, log_path, verbose = True):
        self.log_path =log_path
        self.verbose = verbose
        self.logged_param =  []
        

    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


    def loss_dict_log(self, loss_dict, set_name):
        message = set_name.upper() + " "
        for k, v in loss_dict.items():
            message += f"{k.replace(set_name.upper() + '_', '')} : {v:.5f} "
        return message

    def resister_params(self, params):
        assert type(params) in [str, list], "Invalid params type. It should be single string or list filled with string"

        if isinstance(params, list):
            self.logged_param.extend(params)
        else:
            self.logged_param.append(params)
        

    def log_params(self, params):
        for name, param in params.items():
            if name in self.logged_param:
                message = f"{name.upper()} : {str(param).upper()}"
                self.log(message)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:

    def __init__(self, model, config):
        self.model = model
        self.prep(config)

    def prep(self, config):
        """
        Prepare for fitting
        """
        self.config = config # for save

        # configuration setting except scheduler
        for k, v in config.items():
            if 'schedule' not in k:
                setattr(self, k, v)

        #self.early_stop = 0
        self.best_summary_loss = 10**5

        # path
        self.model_path = os.path.join(config.save_path[0], 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # wandb error issue.. 
        self.logger = Logger(config.log_path)

        if config.model_id == '':
            self.model_id  = f"{type(self.model).__name__}_{datetime.now().strftime('%m%d')}{str(int(datetime.now().strftime('%H')) + 9).zfill(2)}"
        
        else:
            self.model_id = config.model_id


        # log params
        print('Logging Hyper Parameters..')
        self.logger.resister_params(['model_nm', 'batch_size', 'accumulative', 'window'])
        self.logger.log_params(config)

    def loop_indicator(self, e, val_loss):

        if self.best_summary_loss > val_loss:
            # record best 
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e

            # save
            if e > self.save_point:
                self.save(f'{self.model_path}/{self.model_id}_{str(e).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.model_path}/{self.model_id}_*epoch.bin'))[:-1]:
                    os.remove(path)
        else:
            # early stop
            self.early_stop += 1


        if self.early_stop == self.early_stop_rounds:
            return True
        else: 
            return False

    def fit(self, train_loader, valid_loader, use_wandb = True):

        if use_wandb:
            run = wandb.init(
                # Set the project where this run will be logged
                project= self.config.project, 
                # Track hyperparameters and run metadata
                config=self.config)

            for e in range(self.epochs):

                start = time.time()

                train_loss_dict  = self.train_one_epoch(train_loader)
                valid_loss_dict = self.validate(valid_loader)
                
                train_loss_dict.update(valid_loss_dict)

                wandb.log(train_loss_dict)


                if self.warmup_steps <= e:
                    if self.model.scheduler:
                        self.scheduler.step(valid_loss_dict['loss'])

                if self.loop_indicator(e, valid_loss_dict['loss']):
                    run.finish()
                    break
                
            run.finish()
        else:
            
            for e in range(self.epochs):

                start = time.time()

                train_loss_dict  = self.train_one_epoch(train_loader)
                valid_loss_dict = self.validate(valid_loader)

                message = f'[Epoch {e}]'
                message += self.logger.loss_dict_log(train_loss_dict, 'train')
                message += self.logger.loss_dict_log(valid_loss_dict, 'valid')
                message += f'Time : {time.time() - start:.5f} s'

                self.logger.log(message)

                if self.warmup_steps <= e:
                    if isinstance(self.model, nn.DataParallel):
                        if self.model.module.scheduler:
                            self.model.module.scheduler.step(valid_loss_dict['loss'])
                    else:
                        if self.model.scheduler:
                            self.model.scheduler.step(valid_loss_dict['loss'])
                if self.loop_indicator(e, valid_loss_dict['loss']):
                    break
                
            self.logger.log('Finished')           
         


    def train_one_epoch(self, loader):

        self.model.train()

        if isinstance(self.model, nn.DataParallel):
            self.model.module.meter_initialize()
            self.model.module.optimizer.zero_grad()

            for i, batch in enumerate(loader):
                self.model.module.optimize(batch, i)

            return self.model.module.loss_dict()

        else:
            self.model.meter_initialize()
            self.model.optimizer.zero_grad()

    
            for i, batch in enumerate(loader):
                self.model.optimize(batch, i)

        
            return self.model.loss_dict()


    def validate(self, loader):

        self.model.eval()


        if isinstance(self.model, nn.DataParallel):
            self.model.module.meter_initialize()

            for i, batch in enumerate(loader):
                with torch.no_grad():
                    self.model.module.validate(batch)
            return self.model.module.loss_dict()

        else:
            self.model.meter_initialize()

            for i, batch in enumerate(loader):
                with torch.no_grad():
                    self.model.validate(batch)
            return self.model.loss_dict()

    def save(self, path):
        self.model.eval()
        if isinstance(self.model, nn.DataParallel):
            torch.save({
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict' : self.model.module.optimizer.state_dict(),
            'scheduler_state_dict' : self.model.module.scheduler.state_dict(),
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)
        
        else:
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict' : self.model.optimizer.state_dict(),
            'scheduler_state_dict' : self.model.scheduler.state_dict(),
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)

    def load(self, path, load_optimizer = True, load_scheduler = True):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        config = checkpoint['configuration']

        self.prep(config)
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.model.eval()

    def set_lr(self,lr = False, ratio = False):
        if ratio and not lr:
            self.lr *= ratio
        else:
            self.lr = lr
            self.optimizer.param_groups[0]['lr'] = self.lr


