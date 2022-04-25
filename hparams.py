from easydict import EasyDict as edict
import torch

hparams = edict(
    
        batch_size = 4,
        accumulative = 1,
        num_workers = 4,


        weight_decay = 0.001,
        max_grad_norm = 0,
        lr = 0.001,
        early_stop_rounds = 12,
        epochs = 25,
        warmup_steps = 10,
        
        optimizer_cls = None,
        
        step_scheduler = True,
        SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau, # scheduler
        scheduler_params = dict(      # scheduler params
            mode='min',        
            factor=0.2,
            patience= 5 ,
            verbose=True, 
            threshold=0.01,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-12,
            eps=1e-08
        ),        

        save_point = 10,
        save_path = './model', # model save path
        log_path = './log',
        model_id = '',

        verbose = True,
        verbose_step = 1,

        conv_cls = None,
        act_cls = None,
        model_nm = 'eca_nfnet_l0', # backbone feature extractor / nfnet_f0 eca_nfnet_l0 eca_nfnet_l1 ecaresnet50d
        project = "glacier",
        

        loss_fn = None, # nn.BCEWithLogitsLoss nn.MSELoss,


)