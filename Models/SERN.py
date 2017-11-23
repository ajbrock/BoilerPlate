## Wide ResNet with FreezeOut
# Based on code by xternalz: https://github.com/xternalz/WideResNet-pytorch
# WRN by Sergey Zagoruyko and Nikos Komodakis
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# ResNeXt from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
# Squeeze and Excite Modules
class SE(nn.Module):
    def __init__(self,in_channels, r=16):
        super(SE, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_channels,in_channels//r, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels//r, in_channels, 1),
                                 nn.Sigmoid())
    def forward(self,x, shortcut):
        out = self.mlp(F.avg_pool2d(x,(x.size(2),x.size(3))))
        return torch.addcmul(shortcut, x, out.expand_as(x))
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, widen_factor,  cardinality=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D//2)
        self.conv_conv = nn.Conv2d(D//2, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        residual = self.shortcut(x)
        return F.relu(self.se(bottleneck,residual), inplace=True)
    
# note: we call it DenseNet for simple compatibility with the training code.
# similar we call it growthRate instead of widen_factor
class Network(nn.Module):
    def __init__(self, widen_factor, depth, nClasses, epochs, dropRate=0.0, num_ims=0):
        super(Network, self).__init__()       
        
        self.epochs = epochs
        
        nChannels = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor, 512 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        
        self.widen_factor = widen_factor
        
        block = ResNeXtBottleneck
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=7, stride=1,
                               padding=3, bias=False)
        
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
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, self.widen_factor))
          
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
        # if meta is None:
        out = F.log_softmax(self.fc(out))
        # else:
            # i,j,k = meta
            # out = torch.cat([F.linear(out,weight=self.fc.weight[:i],bias=self.fc.bias[:i]),
                             # F.linear(out,weight=self.fc.weight[48:48+j[i]],bias=self.fc.bias[48:48+j]),
                             # F.linear(out,weight=self.fc.weight[:i],bias=self.fc.bias[:i])
        return out