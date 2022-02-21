import torch
import torch.nn as nn
from torch.nn import functional as F

from funcs import *
from utils import *

# code : https://github.com/pytorch/pytorch/issues/1333

class Casual_Conv(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Casual_Conv, self).__init__(*args, **kwargs, padding = 0)
        self.__padding = (kwargs['kernel_size'] - 1) * kwargs['dilation']
        
    def forward(self, x):
      
        return super(Casual_Conv, self).forward(F.pad(x, (self.__padding, 0)))

