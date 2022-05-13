import random
import numpy as np
import torch

a = torch.randn(5)
b = torch.randn(5)
print(torch.mean(torch.eq(a,b).float()))