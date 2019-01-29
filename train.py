#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
SMASH Training Function
Andy Brock, 2017

This script trains and tests a SMASH network, or a resulting network.

Based on Jan Schlüter's DenseNet training code:
https://github.com/Lasagne/Recipes/blob/master/papers/densenet
'''

import os
import logging
import sys
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from utils import  get_data_loaders, MetricsLogger, progress, nClass_dict
import utils

def train_parser():
  usage = 'Basic boilerplate code.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '-D', '--depth', type=int, default=40,
    help='Reference Network depth in layers/block (default: %(default)s)')
  parser.add_argument(
    '-k', '--width', type=int, default=4,
    help='Reference network widening factor (default: %(default)s)')
  parser.add_argument(
    '--dataset', type=str, default='C100',
    help='Which Dataset to train on, out of C10, C100, MN10, MN40, STL10 (default: %(default)s)')   
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  parser.add_argument(
    '--augment', action='store_true', default=True,
    help='Perform data augmentation (enabled by default)')
  parser.add_argument(
    '--no-augment', action='store_false', dest='augment',
    help='Disable data augmentation')
  parser.add_argument(
    '--validate', action='store_true', default=True,
    help='Perform validation on validation set (default: %(default)s)')
  parser.add_argument(
    '--no-validate', action='store_false', dest='validate',
    help='Disable validation')
  parser.add_argument(
    '--test', action='store_true', default=False,
     help='Evaluate on test set after every epoch (default: %(default)s)')
  parser.add_argument(
    '--validate-seed', type=int, default=0,
    help='Which random seed to use when selecting a validation set (default: %(default)s)')
  parser.add_argument(
    '--validate-split', type=int, default=10,
    help='What percentage of the data to use as the validation split (default: %(default)s)')
  parser.add_argument(
    '--num-workers', type=int, default=3,
    help='How many workers for the dataloader. (default: %(default)s)')
  parser.add_argument(
    '--epochs', type=int, default=100,
    help='Number of training epochs (default: %(default)s)')
  parser.add_argument(
    '--experiment_name', type=str, default='default_save', metavar='FILE',
    help=('Save network weights to given .pth file'+
          '(default: automatically name based on input args'))    
  parser.add_argument(
    '--batch-size', type=int, default=50,
    help='Images per batch (default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Whether or not to resume training')
  parser.add_argument(
    '--model', type=str, default='WRN', metavar='FILE',
    help='Which model to use')
  parser.add_argument(
    '--fp16', action='store_true', default=False,
    help='Train with half-precision (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--validate-every', type=int, default=1,
    help='Test after every this many epochs (default: %(default)s)')
  parser.add_argument(
    '--duplicate-at-checkpoints', action='store_true', default=False,
    help='Save an extra copy every 5 checkpoints (default: %(default)s)')
  parser.add_argument(
    '--fold', type=int, default=10,
    help='Which STL-10 training fold to use, 10 uses all (default: %(default)s)')
  parser.add_argument(
    '--top5', action='store_true', default=False,
    help='Measure top-5 error on valid/test instead of top-1 (default: %(default)s')
  parser.add_argument(
    '--no-progress', action='store_false', default=True, dest='progbar',
    help='Do not use progress bars (good for running on remote servers) (default: %(default)s)')
  return parser

def run(config):
    
  # Number of classes, add it to the config
  config['num_classes'] = utils.num_class_dict[config['dataset']]

  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # choose device; by default (and unless this code is modified) it's GPU 0
  device='cuda:0'
  
  # Create logs and weights folder if they don't exist
  if not os.path.exists('logs'):
    os.mkdir('logs')
  if not os.path.exists('weights'):
    os.mkdir('weights')
      
  # Name of the file to which we're saving losses and errors.
  experiment_name = utils.name_from_config(config) if config['experiment_name'] is not 'default' else config['experiment_name']

  # Prepare metrics logging
  metrics_fname = 'logs/' + experiment_name + '_log.jsonl'
  print('Metrics will be saved to {}'.format(metrics_fname))
  mlog = utils.MetricsLogger(metrics_fname, reinitialize=(not config['resume']))
  
  # Import the model module
  sys.path.append('models')
  model_module = __import__(config['model'])#.strip('.py')) # remove accidental.py

  # Build network, either by initializing it or re-lodaing.
  net = model_module.Network(**config)
  net = net.to(device)
  print(net)
  if config['fp16']:
    net = net.half()
  
  
  # prepare the script state dict
  state_dict={'epoch': 0, 'batch_size': config['batch_size']}
  
  if config['resume']:
    print('loading network ' + experiment_name + '...')
    net = torch.load('weights/%s.pth' % experiment_name)
    # net.load_state_dict('weights/%s.pth' % experiment_name)
    # net.optim.load_state_dict('weights/%s_optim.pth' % experiment_name)
    
  # Prepare the run_net
  if config['parallel']:
    print('Parallelizing net...')
    run_net = nn.DataParallel(net)
  else:
    run_net = net
    
  start_epoch = net.epoch + 1 if hasattr(net, 'epoch') else 0 
  print('Number of params: {}'.format(
               sum([p.data.nelement() for p in net.parameters()]))
               )
  # If not validating, set val split to 0
  if not config['validate']:
    config['validate_split'] = 0
  # Get information specific to each dataset
  loaders = utils.get_data_loaders(config['dataset'], config['augment'], config['validate'], config['test'],
                             config['batch_size'],  config['fold'], config['validate_seed'],
                             config['validate_split']/100., config['num_workers'])
  train_loader = loaders[0]
  
  if config['validate']:
      val_loader = loaders[1]
  if config['test']:
      test_loader = loaders[2]
      
  # Training Function, presently only returns training loss
  # x: input data
  # y: target labels
  def train_fn(x, y):
    net.optim.zero_grad()   
    output = run_net(x)
    loss = F.nll_loss(output, y)
    training_error = output.data.max(1)[1].ne(y).sum()    
    loss.backward()
    net.optim.step()
    return loss.data.item(), training_error

  # Testing function, returns test loss and test error for a batch
  # x: input data
  # y: target labels
  def test_fn(x, y):
    with torch.no_grad():
      output = run_net(x)            
      test_loss = F.nll_loss(output, y)

      # If we're running Imagenet, we may want top-5 error:
      if config['top5']:
        top5_preds = np.argsort(output.data.cpu().numpy())[:,:-6:-1]
        test_error = len(y) - np.sum([np.any(top5_i == y_i) for top5_i, y_i in zip(top5_preds,y)])
      else:
        # Get the index of the max log-probability as the prediction.
        pred = output.data.max(1)[1]
        test_error = pred.ne(y).sum()

      return test_loss.data.item(), test_error

  # Finally, launch the training loop.
  print('Starting training at epoch '+str(start_epoch)+'...')
  for epoch in range(start_epoch, config['epochs']):
    # Pin the current epoch on the network.
    net.epoch = epoch

    # shrink learning rate at scheduled intervals, if desired
    if 'epoch' in net.lr_sched and epoch in net.lr_sched['epoch']:

      print('Annealing learning rate...')

      # Optionally checkpoint at annealing
      if net.checkpoint_before_anneal:
          torch.save(net,'weights/' + str(epoch) + '_' + experiment_name + '.pth')

      for param_group in net.optim.param_groups:
          param_group['lr'] *= 0.1

    # List where we'll store training loss
    train_loss, train_err = [], []

    # Prepare the training data
    if config['progbar']:
        batches = utils.progress(
            train_loader, desc='Epoch %d/%d, Batch ' % (epoch + 1, config['epochs']),
            total=len(train_loader.dataset) // config['batch_size'])
    else:
        batches = train_loader

    # Put the network into training mode
    net.train()

    # Execute training pass
    for x, y in batches:      
      # Update LR if using cosine annealing
      if 'itr' in net.lr_sched:
        net.update_lr(max_j=config['epochs'] * len(train_loader.dataset) // config['batch_size'])
          
      loss, err = train_fn(x.to(device), y.to(device))
      train_loss.append(loss)
      train_err.append(err)

    # Report training metrics
    train_loss = float(np.mean(train_loss))
    train_err = 100 * float(np.sum(train_err)) / len(train_loader.dataset)
    print('  training loss:\t%.6f, training error: \t%.2f%%' % (train_loss, train_err))
    mlog.log(epoch=epoch, train_loss=train_loss, train_err=train_err)

    # Optionally, take a pass over the validation set.
    if config['validate'] and not ((epoch+1) % config['validate_every']):

      # Lists to store
      val_loss, val_err = [], []

      # Set network into evaluation mode
      net.eval()

      # Execute validation pass
      if config['progbar']:
        batches = tqdm(val_loader)
      else:
        batches = val_loader
          
      for x, y in batches:
        loss, err = test_fn(x.to(device), y.to(device))
        val_loss.append(loss)
        val_err.append(err)

      # Report validation metrics
      val_loss = float(np.mean(val_loss))
      val_err =  100 * float(np.sum(val_err)) / len(val_loader.dataset)
      print('  validation loss:\t%.6f,  validation error:\t%.2f%%' % (val_loss, val_err))
      mlog.log(epoch=epoch, val_loss=val_loss, val_err=val_err)

    # Optionally, take a pass over the validation or test set.
    if config['test'] and not ((epoch+1) % config['validate_every']):
      # Lists to store
      test_loss, test_err = [], []

      # Set network into evaluation mode
      net.eval()

      # Execute validation pass
      if config['progbar']:
        batches = tqdm(test_loader)
      else:
        batches = test_loader
          
      for x, y in batches:
        loss, err = test_fn(x.to(device), y.to(device))
        test_loss.append(loss)
        test_err.append(err)

      # Report validation metrics
      test_loss = float(np.mean(test_loss))
      test_err =  100 * float(np.sum(test_err)) / len(test_loader.dataset)
      print('  test loss:\t%.6f,  test error:\t%.2f%%' % (test_loss, test_err))
      mlog.log(epoch=epoch, test_loss=test_loss, test_err=test_err)
    # Save weights for this epoch        
    if type(net) is nn.DataParallel:
      print('saving de-parallelized weights to ' + experiment_name + '...')
      torch.save(net.module, 'weights/%s.pth' % experiment_name)
    else:
      print('saving weights to %s...' % experiment_name)
      torch.save(net, 'weights/%s.pth' % experiment_name)
    
    # If requested, save a checkpointed copy with a different name
    # so that we have them for reference later.
      

  # At the end of it all, save weights even if we didn't checkpoint.
  if experiment_name:
    torch.save(net, 'weights/%s.pth' % experiment_name)


def main():
  # parse command line
  parser = train_parser()
  config = vars(parser.parse_args())
  print(config)
  # run; replace this with a for loop to run multiple sequential jobs.
  run(config)


if __name__ == '__main__':
    main()
