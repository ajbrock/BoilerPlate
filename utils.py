#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
Andy's Notes: Need to properly credit things based on where we got them.
To do: enable more in-depth validation splits
'''

from __future__ import print_function
import sys
import os
import time
import json
import logging
import math
import numpy as np

from PIL import Image
from functools import reduce
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
# import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets as dset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

# Dictionary containing number of classes for each dataset
num_class_dict = {'C10': 10,
               'C100': 100,
               'MN40': 40,
               'MN10': 10,
               'I1000': 1000,
               'ILSVRC256': 1000,
               'STl10': 10}


            
# Convenience function to centralize all data loaders
def get_data_loaders(which_dataset, augment=True, validate=True, test=True,
                     batch_size=50, fold='all', 
                     validate_seed=0, validate_split=0.1, num_workers=3):
  # Potentially use a different batch size for testing, since
  # inference memory requirements are lower    
  test_batch_size = batch_size
  
  # Initialize train_transform to be None; by default, it will get 
  # overwritten with our 32x32 transform, but we can also have dataset
  # specific transforms.
  train_transform = None
  test_transform = None
  
  if which_dataset == 'C10':
    print('Loading CIFAR-10...')
    root = 'cifar'
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    dataset = dset.CIFAR10
    test_batch_size = 5 * batch_size

  elif which_dataset == 'C100' or which_dataset == 100:
    print('Loading CIFAR-100...')
    root = 'cifar'
    norm_mean = [0.50707519, 0.48654887, 0.44091785]
    norm_std = [0.26733428, 0.25643846, 0.27615049]
    dataset = dset.CIFAR100
    test_batch_size = 5 * batch_size
      
  elif which_dataset == 'MN40' or which_dataset == 40:
    print('Loading ModelNet-40...')
    root = 'modelnet'
    norm_mean = [0,0,0] # dummy mean
    norm_std = [1,1,1] # dummy std
    dataset = dset.MN40
    test_batch_size = int(np.ceil(batch_size / 10.)) # For parallelism
  elif which_dataset == 'MN10':
    print('Loading ModelNet-10...')
    root = 'modelnet'
    norm_mean = [0,0,0] # dummy mean
    norm_std = [1,1,1] # dummy std
    dataset = dset.MN10
    test_batch_size = int(np.ceil(batch_size / 10.)) # For parallelism
      
  elif which_dataset == 'I1000' or which_dataset == 1000:
    print('Loading 32x32 Imagenet-1000...')
    root = 'imagenet'
    norm_mean = [0.48109809447859192, 0.45747185440340027, 0.40785506971129742]
    norm_std = [0.26040888585626459, 0.25321260169837184, 0.26820634393704579]
    dataset = dset.I1000
  
  elif which_dataset == 'ILSVRC64' or which_dataset == 'ILSVRC128' or which_dataset == 'ILSVRC256' :
    dim = int(which_dataset[6:])
    print('Loading %dx%d Imagenet-1000 for Eddie...' % (dim, dim))
    root = root = '/home/s1580274/scratch/data/ImageNet'
    norm_mean = [0.48109809447859192, 0.45747185440340027, 0.40785506971129742]
    norm_std = [0.26040888585626459, 0.25321260169837184, 0.26820634393704579]
    dataset = dset.ImageFolder
    norm_transform = transforms.Normalize(norm_mean, norm_std)
    train_transform = transforms.Compose([
            RandomCropLongEdge(),
            transforms.Resize(dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_transform
        ])
    test_transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(dim),
            transforms.ToTensor(),
            norm_transform
        ])
  elif which_dataset == 'STL10':
    root = 'STL'
    print('Loading STL-10...')
    norm_mean = [0.4467106206597222, 0.439809839835240, 0.406646447099673]
    norm_std =  [0.2603409782662329, 0.256577273113443, 0.271267381452256]
    dataset = dset.STL10
  
  # Prepare transforms and data augmentation
  norm_transform = transforms.Normalize(norm_mean, norm_std)
  
  if augment:
    print('Data will be augmented...')
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_transform
        ])
  else:
    print('Data will NOT be augmented...')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        norm_transform
    ])
  if test_transform is None:    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm_transform
        ])
  
  kwargs = {'num_workers': num_workers, 'pin_memory': True}

  train_set = dataset(
      root=root,
      train=True,
      download=True,
      transform=train_transform if augment else test_transform,
      validate_seed=validate_seed,
      val_split=validate_split)
  # If we're evaluating on the test set, load the test set
  
  # If we're evaluating on the validation set, prepare validation set
  # as the last 5,000 samples in the training set.
  
  if validate:
    print('Using validation set...')
    np.random.seed(validate_seed)
    val_set = dataset(root=root, train='validate', download=True,
                       transform=test_transform,
                       validate_seed=validate_seed,
                       val_split=validate_split)
    # train_set.data = np.delete(train_set.data, val_set.val_indices,0)
    # train_set.labels = list(np.delete(train_set.labels, val_set.val_indices))
    val_loader = DataLoader(val_set, batch_size=test_batch_size,
                           shuffle=False, **kwargs)
      
  if test:
    print('Using test set...')
    test_set = dataset(root=root, train=False, download=True,
                         transform=test_transform)

  

  # Prepare data loaders
  loaders = []
  train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, **kwargs)
  loaders.append(train_loader)
  
  if validate:
    val_loader = DataLoader(val_set, batch_size=test_batch_size,
                         shuffle=False, **kwargs)
    loaders.append(val_loader)
  else:
    loaders.append(None)
  
  if test:
    test_loader = DataLoader(test_set, batch_size=test_batch_size,
                         shuffle=False, **kwargs)
    loaders.append(test_loader)
  else:
    loaders.append(None)
                           
  return loaders
    
# ''' MetricsLogger originally stolen from VoxNet source code.'''
class MetricsLogger(object):

    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                logging.warn('{} exists, deleting'.format(self.fname))
                # self.fname.remove()

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


# def read_records(fname):
    # """ convenience for reading back. """
    # skipped = 0
    # with open(fname, 'rb') as f:
        # for line in f:
            # if not line.endswith('\n'):
                # skipped += 1
                # continue
            # yield json.loads(line.strip())
        # if skipped > 0:
            # logging.warn('skipped {} lines'.format(skipped))
            
"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA.
"""
def progress(items, desc='', total=None, min_delay=0.1):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                    desc, n+1, total, n / float(total) * 100), end=" ")
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))), end=" ")
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))

  
""" Cropping augmentations """
class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__
        
class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = 0 if size[0] == img.size[0] else np.random.randint(low=0,high=img.size[0] - size[0])
    j = 0 if size[1] == img.size[1] else np.random.randint(low=0,high=img.size[1] - size[1])    
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__
# Get factors of a given number
# Taken from this stackexchange answer:
# https://stackoverflow.com/a/6800214
# I haven't bothered to grok its internals, but it works.
def factors(n):
    assert n>0
    return sorted(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
                
# Simple softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  
# Given a config, generate an experiment name
def name_from_config(config):
  return '_'.join([item for item in 
                  [config['model'],
                  'D' + str(config['depth']),
                  'K' + str(config['width']),
                  'fp16' if config['fp16'] else None,
                  config['dataset'],
                  'seed' + str(config['seed']),
                  'val' if config['validate'] else None,
                  str(config['epochs'])+'epochs'] if item is not None])