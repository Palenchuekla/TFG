import random
import torch
import os
import numpy as np

# Reproducibility
def reproducibility(seed=1234):
  '''
    Fuction to iniatialize every random process related to PyTorch: parameters initialization and batching order.
    Also applies to Numpy.
  '''
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ['PYTHONHASHSEED'] = str(seed)