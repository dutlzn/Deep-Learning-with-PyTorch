import torch.nn as nn
import torch 
m = nn.AvgPool2d((3,3))
input = torch.randn(1, 64, 8, 8)
print(input.size())
output = m(input)
print(output.size())