import os
import torch
import torch.nn as nn
import sys

class BaseModel(nn.Module):
  def name(self):
    return "BaseModel"
