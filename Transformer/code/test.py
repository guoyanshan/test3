import torch
from torch import nn
import torch.nn.functional as F
input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
print(input)
output = F.interpolate(input, scale_factor=2, mode='bicubic')
print(output)
x = F.interpolate(output, scale_factor=0.5, mode='bicubic')
print(x)