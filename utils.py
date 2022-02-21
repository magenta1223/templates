from multiprocessing.spawn import import_main_path


import os
from glob import glob
import wandb
import time
from datetime import datetime

import torch

class Logger:
    def __init__(self, log_path, verbose = True):
        self.log_path =log_path
        self.verbose = verbose
        

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
        '''
        모델 훈련에 필요한 모든 것
        '''
        self.config = config # for save

        # configuration setting
        for k, v in config.items():
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

    def fit(self, train_loader, valid_loader, use_wandb = False):

        if use_wandb:

            run = wandb.init(
                # Set the project where this run will be logged
                project= self.config.project, # 이름 바꾸기
                # Track hyperparameters and run metadata
                config=self.config)

            for e in range(self.epochs):

                start = time.time()

                train_loss_dict  = self.train_one_epoch(train_loader)
                valid_loss_dict = self.validate(valid_loader)
                
                self.logger.loss_dict_log( e, {'train' : train_loss_dict, 'valid' : valid_loss_dict }, start  )

                train_loss_dict.update(valid_loss_dict)

                wandb.log(train_loss_dict)


                if self.warmup_steps <= e:
                    try:
                        self.scheduler.step(valid_loss_dict['loss'])
                    except:
                        assert 1 == 0, 'No scheduler'
                if self.loop_indicator(e, valid_loss_dict['loss']):
                    self.logger.log('Early Stop')
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
                    try:
                        self.scheduler.step(valid_loss_dict['VALID_loss'])
                    except:
                        pass
                        #assert 1 == 0, 'No scheduler'
                if self.loop_indicator(e, valid_loss_dict['VALID_loss']):
                    break
                
            self.logger.log('Finished')           

    def train_one_epoch(self, loader):

        self.model.train()

        self.model.meter_initialize()
        self.model.optimizer.zero_grad()


        for i, batch in enumerate(loader):
            self.model.optimize(batch, i)

        
        return self.model.loss_dict(set_name = 'train')


    def validate(self, loader):

        self.model.eval()
        self.model.meter_initialize()

        for i, batch in enumerate(loader):
            with torch.no_grad():
                self.model.validate(batch)
        return self.model.loss_dict(set_name = 'valid')

    def save(self, path):
        self.model.eval()
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
        self.model.optimizer.load_state_dict(checkpoint['model_state_dict'])
        self.model.scheduler.load_state_dict(checkpoint['model_state_dict'])
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