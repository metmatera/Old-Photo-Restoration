import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

from torch.autograd import Variable
from torch.nn.utils import spectral_norm

class Encoder(nn.Module):
  def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
    super(Encoder, self).__init__()
    self.output_nc = output_nc
    
    model = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
      norm_layer(ngf),
      nn.ReLU(True)
    ]
