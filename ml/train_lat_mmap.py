from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils as utils
import sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision.models as models

from torch.utils import data
from torchvision import datasets, transforms
import glob
import time
import socket
import os
from models import *
from custom_data import *
import psutil

process = psutil.Process(os.getpid())
   
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    loss = nn.MSELoss()

    start_time = time.time()
    cur_time = start_time
    print("Train loader has %d minibatches." % len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        
        print(process.memory_info().rss/(1000*1000*1000.))  #current memory usage in gb

        def closure():
            optimizer.zero_grad()
            output = model(data)
            value = loss(output,target) 
            value.backward()
            return value

        optimizer.step(closure)
        prev_time = cur_time
        cur_time = time.time()
        
        print("Epoch %d Minibatch %d took %f " \
              "Epoch of %d mbs would be %f "\
              "(Avg so far is %f)" % (epoch,
                                      batch_idx,
                                      cur_time - prev_time,
                                      len(train_loader),
                                      (cur_time - prev_time)*len(train_loader), 
                                      (cur_time-start_time)/(batch_idx + 1))) 
        
    print("Epoch %d  time was %f" % (epoch, time.time() - start_time))


def test(args, model, device, test_loader,mean,std):
    model.eval()
    loss_fetch = 0
    loss_completion = 0
    loss_total = 0
    correct = 0
    loss = nn.L1Loss(reduction='sum') 
    total_len = 0

    with torch.no_grad():
        for data, target in test_loader:
            total_len += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss_total += loss( output,target)
            
            norm_output = (output*std + mean)
            norm_target = (target*std + mean)
    
            loss_fetch += loss( norm_output[:,0:1].round(),
                                norm_target[:,0:1].round() ).item()
            
            loss_completion += loss( norm_output[:,1:2].round(),
                                     norm_target[:,1:2].round() ).item()
            

    loss_fetch /= total_len
    loss_completion /= total_len
    loss_total /= total_len
    print('Test fetch {:.4f}, completion {:.4f} NN {:.4f} '.format(loss_fetch,
                                                                   loss_completion,
                                                                   loss_total))
    return loss_total.item();

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--data-name',type=str,help="Dataset file")
parser.add_argument('--stats-name',type=str,help="Stats file")
parser.add_argument('--batch-size', type=int, default=2*32*1024, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs')
parser.add_argument('--resume', type=str,
                    help="Model to resume training from",default=None)
parser.add_argument('--no-cuda', action='store_true', default=False )
parser.add_argument('--test', action='store_true', default=False, )
parser.add_argument('--save-model', action='store_true', default=False)

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--rows',type=int);
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    fname = args.data_name
    statsname = args.stats_name


              
    use_cuda = True
   

    print("Batch sz is %d. Save? %s. Cuda? %s. Dev count: %d \nFname %s\n" % 
          (args.batch_size,
           str(args.save_model),
           str(use_cuda),
           torch.cuda.device_count(),fname))


    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: " ,device)
    kwargs = {'num_workers': args.workers, # pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
              'pin_memory': True} if use_cuda else {}

    print("Beginning to load %s" % fname)

    
    stats = np.load(statsname)

    mean = torch.FloatTensor(stats['all_mean'][[1,3]]) # fetch, completion
    std = torch.FloatTensor(np.sqrt(stats['all_var'][[1,3]])) # fetch, completion
    print("mean shape %s std shape %s" % ( str(mean.shape), str(std.shape) ))
    
    mean, std = mean.to(device), std.to(device)

    mmapped_arr = np.memmap(fname, dtype=np.float32,
                            mode='r',shape=(args.rows,context_length*inst_length)) #(5*139*259717,3666))
    
    print(mmapped_arr.shape)
 
    data_set = MemoryMappedDataset(mmapped_arr)   
    
    if args.resume:
        print("RESUMING")
        model = torch.load( args.resume,map_location='cpu' )
        model.eval()
    else:
        print("STARTING FROM SCRATCH")
        model = CNN3(2, 5, 64, 5, 64, 5, 256, 400)

        #model = CNN3_P_D_C(out = 2,        
        #                   pc = 64, 
        #                   ck1 = 5,
        #                   ch1 = 64, 
        #                   ck2 = 5, 
        #                   ch2 = 64,
        #                   ck3= 5, 
        #                   ch3 = 256, 
        #                   f1 = 400)



    model = nn.DataParallel(model)
    model.to(device)
    
    if args.test:
        all_model_names=list(glob.iglob("models/*.pt"))
        all_model_names.sort(key=os.path.getmtime)

        for model_name in all_model_names:
              model= torch.load( model_name, map_location='cuda' if use_cuda else 'cpu')
              model.to(device)
              data_loader = data.DataLoader(data_set, 
                                            batch_size=args.batch_size,
                                            **kwargs)

              test(args=args, 
                   model=model, 
                   device=device,
                   test_loader=data_loader,
                   mean=mean,
                   std=std)
        return
              
    data_loader = data.DataLoader(data_set, 
                                  shuffle=False, # True
                                  batch_size=args.batch_size,
                                  **kwargs)

    test_data_loader = data.DataLoader(data_set, 
                                       shuffle=False,
                                       batch_size=args.batch_size,
                                       **kwargs)

    optimizer = optim.Adam(model.parameters())
    
    best, best_idx = [float('inf'),-1]

    multiplier=10

    for epoch in range(1, args.epochs + 1):
        train(args=args,
              model=model,
              device=device, 
              train_loader=data_loader,
              optimizer=optimizer,
              epoch=epoch)
        if (epoch % multiplier == 0):
              t = test(args=args, 
                       model=model, 
                       device=device,
                       test_loader=test_data_loader,
                       mean=mean,
                       std=std)
              if t < best:
                  print("Got improvement at {} (from {:.4f} to {:.4f}".format(epoch,best,t))
                  best , best_idx = (t,epoch)
                  if use_cuda:
                      torch.save(model.module, "models/best.pt" )
                  else:
                      torch.save(model, "models/best.pt" )
                      
        if (args.save_model) and (epoch % multiplier == 0):
            print("Saving")
            if use_cuda:
                torch.save(model.module,
                           "models/%d.pt" % epoch)
            else:
                torch.save(model,
                           "models/%d.pt" % epoch)

        
if __name__ == '__main__':
    main()
