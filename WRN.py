## Wide ResNet with FreezeOut
# Based on code by xternalz: https://github.com/xternalz/WideResNet-pytorch
# WRN by Sergey Zagoruyko and Nikos Komodakis
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
    def forward(self, x):
               
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        
        return out
    
# note: we call it DenseNet for simple compatibility with the training code.
# similar we call it growthRate instead of width
class Network(nn.Module):
  def __init__(self, width, depth, num_classes, epochs, dropRate=0.0, **kwargs):
    super(Network, self).__init__()       
    
    self.epochs = epochs
    
    nChannels = [16, 16*width, 32*width, 64*width]
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
    self.fc = nn.Linear(nChannels[3], num_classes)
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
                           weight_decay=5e-4)
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
    out = torch.mean(out.view(out.size(0),out.size(1),-1),2)#F.avg_pool2d(out, (out.size(2),out.size(3)))
    out = out.view(-1, self.nChannels)
    return F.log_softmax(self.fc(out), -1)