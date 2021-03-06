import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(True),
        nn.Conv3d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(True),
        nn.Conv3d(in_channel, out_channel, 1),
        nn.AvgPool3d(2, 2)
    )
    return trans_layer


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
            
        self.net = nn.Sequential(*block)


    #前向传播 过程
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

class densenet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        in_channel = 3 
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channel, 64, 3, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(3, 2, padding=1)
        )
        
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition(channels, channels // 2)) # 通过 transition 层将大小减半，通道数减半
                channels = channels // 2
        
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm3d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool3d(3))
        
        self.classifier = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = densenet(num_classes=101)

    outputs = net.forward(inputs)
    print(outputs.size())