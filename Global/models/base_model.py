import os
import torch
import torch.nn as nn
import sys

class BaseModel(nn.Module):
  def name(self):
    return "BaseModel"
  
  def initialize(self, opt):
    self.opt = opt
    self.gpu_ids = opt.gpu_ids
    self.isTrain = opt.isTrain
    self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
    self.save_dir = os.path.join(opt_checkpoints_dir, opt.name)
    
  def set_input(self, input):
    self.input = input
   
  def forward(self):
    pass
  
  def test(self):
    pass
  
  def get_image_paths(self):
    pass
  
  def optimize_parameters(self):
    pass
  
  def get_current_visuals(self):
    return self.input
  
  def get_current_errors(self):
    return {}
  
  def save(self, label):
    pass
  
  def update_learning_rate():
    pass
