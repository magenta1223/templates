import torch
import numpy as np

# Calculate symmetric padding for a convolution from timm
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """
    calculate padding for preserving resolution of 2d image 
    """
    
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

