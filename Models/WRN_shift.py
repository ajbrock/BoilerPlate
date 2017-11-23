## Wide ResNet with Shift and incorrect hyperparams.
# Based on code by xternalz: https://github.com/xternalz/WideResNet-pytorch
# WRN by Sergey Zagoruyko and Nikos Komodakis
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim

import numpy as np


#torch.cat([torch.zeros(x.size(0),self.channels_per_group,1,x.size(2)).cuda()
# We'll allocate any leftover channels to the center group
class shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size**2)
        # self.groups = self.in_planes // kernel_size
    
    # Leave the final group in place
    # We've actually reversed the tops+bottoms vs left+right (first spatial index being rows, second being columns). Oh well.
    def forward(self,x):
        out = V(torch.zeros(x.size()).cuda())
        # Alias for convenience
        cpg = self.channels_per_group
        # Bottom shift, grab the Top element
        i=0
        out[:, i * cpg : (i + 1) * cpg, 1:, :] = x[:, i * cpg : (i + 1) * cpg, :-1, :]
        out[:, i * cpg : (i + 1) * cpg, 0, :] = 0
        
        # Top shift, grab the Bottom element
        i=1
        out[:, i * cpg : (i + 1) * cpg, :-1, :] = x[:, i * cpg : (i + 1) * cpg, 1:, :]
        out[:, i * cpg : (i + 1) * cpg, -1, :] = 0
        
        
        # Right shift, grab the left element 
        i=2
        out[:, i * cpg : (i + 1) * cpg, :, 1:] = x[:, i * cpg : (i + 1) * cpg, :, :-1]
        out[:, i * cpg : (i + 1) * cpg, :, 0] = 0
        
        
        # Left shift, grab the right element
        i=3
        out[:, i * cpg : (i + 1) * cpg, :, :-1] = x[:, i * cpg : (i + 1) * cpg, :, 1:]
        out[:, i * cpg : (i + 1) * cpg, :, -1] = 0
        
        # Bottom Right shift, grab the Top left element 
        i=4
        out[:, i * cpg : (i + 1) * cpg, 1:, 1:] = x[:, i * cpg : (i + 1) * cpg, :-1, :-1]
        out[:, i * cpg : (i + 1) * cpg, 0, :] = 0
        out[:, i * cpg : (i + 1) * cpg, :, 0] = 0
        
        # Bottom Left shift, grab the Top right element
        i=5
        out[:, i * cpg : (i + 1) * cpg, 1:, :-1] = x[:, i * cpg : (i + 1) * cpg, :-1, 1:]
        out[:, i * cpg : (i + 1) * cpg, 0, :] = 0
        out[:, i * cpg : (i + 1) * cpg, :, -1] = 0
        
        # Top Right shift, grab the Bottom Left element
        i=6
        out[:, i * cpg : (i + 1) * cpg, :-1, 1:] = x[:, i * cpg : (i + 1) * cpg, 1:, :-1]
        out[:, i * cpg : (i + 1) * cpg, -1, :] = 0
        out[:, i * cpg : (i + 1) * cpg, :, 0] = 0
        
        # Top Left shift, grab the Bottom Right element
        i=7
        out[:, i * cpg : (i + 1) * cpg, :-1, :-1] = x[:, i * cpg : (i + 1) * cpg, 1:, 1:]
        out[:, i * cpg : (i + 1) * cpg, -1, :] = 0
        out[:, i * cpg : (i + 1) * cpg, :, -1] = 0
        

        return out
        
        
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate,E=9):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        
        self.conv2 = shift(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
    def forward(self, x):
               
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv2(self.conv1(out if self.equalInOut else x))))
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv3(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        # print(x.size(),out.size())
        return out
    
# note: we call it DenseNet for simple compatibility with the training code.
# similar we call it growthRate instead of widen_factor
class Network(nn.Module):
    def __init__(self, widen_factor, depth, nClasses, epochs, dropRate=0.0):
        super(Network, self).__init__()       
        
        self.epochs = epochs
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)

        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 1st block
        self.block1 = self._make_layer(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = self._make_layer(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = self._make_layer(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], nClasses)
        self.nChannels = nChannels[3]
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
            
                
        # Optimizer
        self.lr = 1e-1
        self.optim = optim.SGD(params=self.parameters(),lr=self.lr,
                               nesterov=True,momentum=0.9, 
                               weight_decay=1e-4)
        # Iteration Counter            
        self.j = 0  

        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise. 
        self.lr_sched = {'itr':0}
    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):        
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
          
        return nn.Sequential(*layers)
    
    def update_lr(self, max_j):        
        for param_group in self.optim.param_groups:
            param_group['lr'] = (0.5 * self.lr) * (1 + np.cos(np.pi * self.j / max_j))
        self.j += 1 
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, (out.size(2),out.size(3)))
        out = out.view(-1, self.nChannels)
        return F.log_softmax(self.fc(out))