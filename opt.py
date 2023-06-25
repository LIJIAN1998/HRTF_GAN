from functools import partial
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

print("import done!")
print("using cuda? ", torch.cuda.is_available())