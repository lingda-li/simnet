from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils as utils
import sys
from torch.utils import data
from torchvision import datasets, transforms
import glob
import time
import socket
import os
from models import *

suffix = None

def print_file(msg):
    with open("output_" + str(suffix), "a") as myfile:
        myfile.write(str(msg) + '\n')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print("Device:" ,device)
    loss = nn.MSELoss() 
#    loss = nn.L1Loss() #(reduction=None) 

    start_time = time.time()
    cur_time = start_time
    #print("Train loader has %d minibatches." % len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(data)
            value = loss(output, target)
            value.backward()
            return value

        #commentd out for lbfgs
        """optimizer.zero_grad()
        print("Forwarding model")
        output = model(data)
        value = loss(output,target) #/(target + 0.01)
        value.backward()"""

        optimizer.step(closure)
        prev_time = cur_time
        cur_time = time.time()
        """ print("Epoch %d Minibatch %d took %f " \
              "Epoch of %d mbs would be %f "\
              "(Avg so far is %f)" % (epoch,
                                      batch_idx,
                                      cur_time - prev_time,
                                      len(train_loader),
                                      (cur_time - prev_time)*len(train_loader), 
                                      (cur_time-start_time)/(batch_idx + 1))) """
        
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
    print('Test set average loss fetch (cycles) {:.4f},  completion (cycles) {:.4f} NN obj {:.4f} '.format(loss_fetch,loss_completion,loss_total))
    return loss_total.item();

def main():
    global print
    global suffix

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--resume', type=str, 
                        help="Model to resume training from",default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test a model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()


    fname = "totalall.npz"
    statsname = "statsall.npz"
    
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    print("Batch sz is %d. Save? %s. Cuda? %s. Dev count: %d \nFname %s\n" % 
          (args.batch_size,
           str(args.save_model),
           str(use_cuda),
           torch.cuda.device_count(),fname))


    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: " ,device)
    kwargs = {'num_workers': 4, 
              'pin_memory': True} if use_cuda else {}

    print("Beginning to load %s" % fname)

    complete_data = np.load(fname)
    stats = np.load(statsname)
    
    feedback_inputs, feedback_targets = None,None
    feedback_input_file = "/raid/data/tflynn/gccvec/feedback_inputs.txt"
    feedback_target_file = "/raid/data/tflynn/gccvec/feedback_targets.txt"
    if os.path.exists(feedback_input_file) and os.path.exists(feedback_target_file):
        feedback_inputs = np.genfromtxt(feedback_input_file);
        feedback_targets = np.genfromtxt(feedback_target_file);
    
        print(feedback_inputs.shape)
        print(feedback_targets.shape)
#    sys.exit()

    mean = torch.FloatTensor(stats['all_mean'][[1,3]]) # fetch, completion
    std = torch.FloatTensor(np.sqrt(stats['all_var'][[1,3]])) # fetch, completion
    print("mean shape %s std shape %s" % ( str(mean.shape), str(std.shape) ))
    
    mean, std = mean.to(device), std.to(device)

    x = complete_data['x']
    y = np.copy(x[:,1:2]) # x[0] is fetch classification data
    y2 = np.copy(x[:,3:4]) # x[2] is completion classification
    y = np.concatenate((y, y2), axis=1) #target is the fetch and completion time
    x[:,0:4] = 0 #setting the target data to zero now. 

    print("X and y shapes:")
    print(x.shape) # 
    print(y.shape) # 

    x = torch.FloatTensor(x)#[:-1000000])
    y = torch.FloatTensor(y)#[:-1000000])
     

    if False: #feedback_inputs is not None and feedback_targets is not None:
        feedback_inputs = torch.FloatTensor(feedback_inputs)
        feedback_targets = torch.FloatTensor(feedback_targets)
        feedback_targets /= std.cpu()
        x = torch.cat((x,feedback_inputs))
        y = torch.cat((y,feedback_targets))
        print("Added in feedback data")
    else:
        print("No feedback being added")
        
    print(x.shape,y.shape)
    #    print(y)
    #print(feedback_targets)
    #sys.exit()

    #print(stats.keys())
    
    print("Completed loading. Shape is %s" % str(x.shape))

    data_set = data.TensorDataset(x,y)   
    
    if args.resume:
        print("RESUMING")
        model = torch.load( args.resume,map_location='cpu' )
        model.eval()
    else:
        print("STARTING FROM SCRATCH")
        model = CNN3_P_D_C(out = y.shape[1],   #MLP
                           pc = 64, 
                           ck1 = 5,
                           ch1 = 64, 
                           ck2 = 5, 
                           ch2 = 64,
                           ck3= 5, 
                           ch3 = 256, 
                           f1 = 400)

    #model = CNN3(1, 5, 64, 5, 64, 5, 256, 400)
    print(type(model))
    if use_cuda:
        model = nn.DataParallel(model)
    model.to(device)
    
    if args.test:
        all_model_names=list(glob.iglob("models/*.pt"))
        all_model_names.sort(key=os.path.getmtime)
        #print("Model names:" + "\n".join(all_model_names))
        for model_name in all_model_names:
              model = torch.load( model_name, map_location='cuda' if use_cuda else 'cpu')
              #model = nn.DataParallel(model)
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
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  **kwargs)

    test_data_loader = data.DataLoader(data_set, 
                                       shuffle=False,
                                       batch_size=args.batch_size,
                                       **kwargs)

    #optimizer = optim.LBFGS(model.parameters()) 
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    best, best_idx = [float('inf'),-1]

    for epoch in range(1, args.epochs + 1):
        train(args=args, 
              model=model,
              device=device, 
              train_loader=data_loader,
              optimizer=optimizer,
              epoch=epoch)
        multiplier=100
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
                      torch.save(model.module, "best.pt" )
                  else:
                      torch.save(model, "best.pt" )
                    
        if epoch > best_idx + 3*multiplier : #e.g. best is at 100, we exit at 401
            print("Ending at {} as there's been no improvement in 300 epochs. Best was at {}".format(epoch,best_idx))
            break

        if (args.save_model) and (epoch % 100 == 0):
            print("Saving")
            if use_cuda:
                torch.save(model.module,
                           "models/%d.pt" % epoch)
            else:
                torch.save(model,
                           "models/%d.pt" % epoch)

        
if __name__ == '__main__':
    main()
