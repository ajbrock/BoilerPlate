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

# Fixup (formerly ZeroInit) by Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma (ICLR 2019)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, layer_index):
        super(BasicBlock, self).__init__()        
        
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)       

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        
        # Scalar gain and bias
        self.gain, self.biases = P(torch.ones(1,1,1,1)), nn.ParameterList([P(torch.zeros(1,1,1,1)) for _ in range(4)])
        # layer index
        self.layer_index = layer_index
        
    def shortcut(self,x):
      if self.convShortcut is not None:
        return self.convShortcut(self.activation(x))
      else:
        return x
    def residual(self, x):
      out = x + self.biases[0]
      out = self.conv1(out) + self.biases[1]
      out = self.activation(out) + self.biases[2]
      out = self.gain * self.conv2(out) + self.biases[3]
      return out
    def forward(self, x):
      return self.residual(x) + self.shortcut(x)
    

class Network(nn.Module):
  def __init__(self, width, depth, num_classes, epochs, **kwargs):
    super(Network, self).__init__()       
    
    self.epochs = epochs
    
    nChannels = [16, 16*width, 32*width, 64*width]
    assert((depth - 4) % 6 == 0)
    # number of layers in a block
    n = int((depth - 4) / 6)

    block = BasicBlock
    
    # 1st conv before any network block
    self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                           padding=1, bias=False)
    
    # Layer index, now only used to count resblocks
    self.layer_index = 0
    # 1st block
    self.block1 = self._make_block(n, nChannels[0], nChannels[1], block, 1)
    # 2nd block
    self.block2 = self._make_block(n, nChannels[1], nChannels[2], block, 2)
    # 3rd block
    self.block3 = self._make_block(n, nChannels[2], nChannels[3], block, 2)
    # global average pooling and classifier
    self.bn1 = nn.BatchNorm2d(nChannels[3])
    self.relu = nn.ReLU(inplace=True)
    self.fc = nn.Linear(nChannels[3], num_classes)
    self.nChannels = nChannels[3]
    
    self.init_weights()   
    
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
  def _make_block(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):        
    resblocks = []
    for i in range(nb_layers):
      resblocks.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, self.layer_index))
      self.layer_index += 1
      
    return nn.Sequential(*resblocks)
  
  def init_weights(self):
    # init fc to zero
    self.fc.weight.data.zero_()
    self.fc.bias.data.zero_()
    
    for block in [self.block1, self.block2, self.block3]:
      for b in block:
        # He init, rescaled by Fixup multiplier
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        print(b.layer_index, math.sqrt(2. / n), self.layer_index **(-0.5))
        b.conv1.weight.data.normal_(0,(self.layer_index ** (-0.5)) *  math.sqrt(2. / n)) 
        b.conv2.weight.data.zero_()
        if b.convShortcut is not None:
          n = b.convShortcut.kernel_size[0] * b.convShortcut.kernel_size[1] * b.convShortcut.out_channels
          b.convShortcut.weight.data.normal_(0, math.sqrt(2. / n)) 
        
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
    out = torch.mean(out.view(out.size(0),out.size(1),-1),2)
    out = out.view(-1, self.nChannels)
    return F.log_softmax(self.fc(out), -1)
