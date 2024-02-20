
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(object):

  def __init__(self,
               time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE

  def forward(self, x):
    # WRITE CODE HERE
    
    return x
  
# # WRITE CODE HERE

class cnn(object):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # WRITE CODE HERE
        
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        # WRITE CODE HERE
        
        # Layers
        # WRITE CODE HERE
        
        return x
