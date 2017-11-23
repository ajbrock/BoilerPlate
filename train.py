#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
PyTorch CNN Training Function
Andy Brock, 2017

This script trains and tests a network on CIFAR-100.

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
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import  get_data_loaders, MetricsLogger, progress, nClass_dict

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
        '--which-dataset', type=str, default='C100',
        help='Which Dataset to train on, out of C10, C100, I1000, MN10, MN40, STL10 (default: %(default)s)')   
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
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--weights-fname', type=str, default='default_save', metavar='FILE',
        help=('Save network weights to given .pth file'+
              '(default: automatically name based on input args'))    
    parser.add_argument(
        '--batch-size', type=int, default=50,
        help='Images per batch (default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Whether or not to resume training')
    parser.add_argument(
        '--model', type=str, default='models/WRN', metavar='FILE',
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
    return parser


# Set the recursion limit to avoid problems with deep nets
sys.setrecursionlimit(5000)

def train_test(depth, width, which_dataset, seed, augment, validate, test,
               validate_seed, validate_split,
               epochs, weights_fname, batch_size, resume, model,  fp16, 
               parallel, validate_every, duplicate_at_checkpoints, fold, top5):
    
   
    # Number of classes
    nClasses = nClass_dict[which_dataset]

    # Seed RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create logs and weights folder if they don't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('weights'):
        os.mkdir('weights')
    
    # Name of the file to which we're saving losses and errors.
    if weights_fname is 'default_save':
        weights_fname = '_'.join([item for item in 
                                [model,
                                'D' + str(depth),
                                'K' + str(width),
                                'fp16' if fp16 else None,
                                which_dataset,
                                'seed' + str(seed),
                                'val' if validate else None,
                                str(epochs)+'epochs'] if item is not None])
    
    # Prepare metrics logging
    metrics_fname = 'logs/' + weights_fname + '_log.jsonl'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    mlog = MetricsLogger(metrics_fname, reinitialize=(not resume))
    
    # Import the model module
    sys.path.append('models')
    model_module = __import__(model)

    # Build network, either by initializing it or re-lodaing.
    if resume:
        logging.info('loading network ' + weights_fname + '...')
        net = torch.load('weights/'+weights_fname + '.pth')

        if parallel:
            net = torch.nn.DataParallel(net)
        if fp16:
            net = net.half()
            
        # Which epoch we're starting from
        start_epoch = net.epoch + 1 if hasattr(net, 'epoch') else 0
        
        # Rescale iteration counter if batchsize requires it.
        if hasattr(net,'j'):
            net.j = int(net.j * net.batch_size / float(batch_size))
            net.batch_size = batch_size

    else:
        net = model_module.Network(width, depth,
                                    nClasses=nClasses,
                                    epochs=epochs)
        net = net.cuda()
        net.batch_size = batch_size
        
        if fp16:
            net = net.half()
        if parallel:
            net = torch.nn.DataParallel(net)
            net.lr_sched = net.module.lr_sched
            net.update_lr = net.module.update_lr
            net.optim = net.module.optim
        
        start_epoch = 0
   
    logging.info('Number of params: {}'.format(
                 sum([p.data.nelement() for p in net.parameters()]))
                 )
    # Get information specific to each dataset
    loaders = get_data_loaders(which_dataset, augment, validate, test,
                               batch_size,  fold, validate_seed,
                               validate_split/100.)
    train_loader = loaders[0]
    
    if validate:
        val_loader = loaders[1]
    if test:
        test_loader = loaders[2]
    # Training Function, presently only returns training loss
    # x: input data
    # y: target labels
    def train_fn(x, y):
        net.optim.zero_grad()
        input = V(x.cuda().half()) if fp16 else V(x.cuda())
    
        output = net(input)

        loss = F.nll_loss(output, V(y.cuda()))
        training_error = output.data.max(1)[1].cpu().ne(y).sum()
        
        loss.backward()
        net.optim.step()
        return loss.data[0], training_error

    # Testing function, returns test loss and test error for a batch
    # x: input data
    # y: target labels
    def test_fn(x, y):
    
        input = V(x.cuda().half(), volatile=True) if fp16 else V(x.cuda(), volatile=True)
        output = net(input)            
        test_loss = F.nll_loss(output, V(y.cuda(), volatile=True)).data[0]

        # If we're running Imagenet, we may want top-5 error:
        if top5:
            top5_preds = np.argsort(output.data.cpu().numpy())[:,:-6:-1]
            test_error = len(y) - np.sum([np.any(top5_i == y_i) for top5_i, y_i in zip(top5_preds,y)])
        else:
            # Get the index of the max log-probability as the prediction.
            pred = output.data.max(1)[1].cpu()
            test_error = pred.ne(y).sum()

        return test_loss, test_error

    # Finally, launch the training loop.
    logging.info('Starting training at epoch '+str(start_epoch)+'...')
    for epoch in range(start_epoch, epochs):

        # Pin the current epoch on the network.
        net.epoch = epoch

        # shrink learning rate at scheduled intervals, if desired
        if 'epoch' in net.lr_sched and epoch in net.lr_sched['epoch']:

            logging.info('Annealing learning rate...')

            # Optionally checkpoint at annealing
            if net.checkpoint_before_anneal:
                torch.save(net,'weights/' + str(epoch) + '_' + weights_fname + '.pth')

            for param_group in net.optim.param_groups:
                param_group['lr'] *= 0.1

        # List where we'll store training loss
        train_loss, train_err = [], []

        # Prepare the training data
        batches = progress(
            train_loader, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
            total=len(train_loader.dataset) // batch_size)

        # Put the network into training mode
        net.train()

        # Execute training pass
        for x, y in batches:
        
            # Update LR if using cosine annealing
            if 'itr' in net.lr_sched:
                net.update_lr(max_j=epochs * len(train_loader.dataset) // batch_size)
                
            loss, err = train_fn(x, y)
            train_loss.append(loss)
            train_err.append(err)

        # Report training metrics
        train_loss = float(np.mean(train_loss))
        train_err = 100 * float(np.sum(train_err)) / len(train_loader.dataset)
        print('  training loss:\t%.6f, training error: \t%.2f%%' % (train_loss, train_err))
        mlog.log(epoch=epoch, train_loss=train_loss, train_err=train_err)

        # Optionally, take a pass over the validation set.
        if validate and not ((epoch+1) % validate_every):

            # Lists to store
            val_loss, val_err = [], []

            # Set network into evaluation mode
            net.eval()

            # Execute validation pass
            for x, y in tqdm(val_loader):
                loss, err = test_fn(x, y)
                val_loss.append(loss)
                val_err.append(err)

            # Report validation metrics
            val_loss = float(np.mean(val_loss))
            val_err =  100 * float(np.sum(val_err)) / len(val_loader.dataset)
            print('  validation loss:\t%.6f,  validation error:\t%.2f%%' % (val_loss, val_err))
            mlog.log(epoch=epoch, val_loss=val_loss, val_err=val_err)

        # Optionally, take a pass over the validation or test set.
        if test and not ((epoch+1) % validate_every):

            # Lists to store
            test_loss, test_err = [], []

            # Set network into evaluation mode
            net.eval()

            # Execute validation pass
            for x, y in tqdm(test_loader):
                loss, err = test_fn(x, y)
                test_loss.append(loss)
                test_err.append(err)

            # Report validation metrics
            test_loss = float(np.mean(test_loss))
            test_err =  100 * float(np.sum(test_err)) / len(test_loader.dataset)
            print('  test loss:\t%.6f,  test error:\t%.2f%%' % (test_loss, test_err))
            mlog.log(epoch=epoch, test_loss=test_loss, test_err=test_err)
        # Save weights for this epoch
        print('saving weights to ' + weights_fname + '...')
        torch.save(net, 'weights/' + weights_fname + '.pth')
        
        # If requested, save a checkpointed copy with a different name
        # so that we have them for reference later.
        if duplicate_at_checkpoints and not epoch%5:
            torch.save(net, 'weights/' + weights_fname + '_e' + str(epoch) + '.pth')

    # At the end of it all, save weights even if we didn't checkpoint.
    if weights_fname:
        torch.save(net, 'weights/' + weights_fname + '.pth')


def main():
    # parse command line
    parser = train_parser()
    args = parser.parse_args()
    print(args)
    # run; replace this with a for loop to run multiple sequential jobs.
    train_test(**vars(args))


if __name__ == '__main__':
    main()
