import os
import sys
import math
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from cfg import *
from utils import *
from models import *


def rank_train(rank, world_size, f, fs, models, modelnum, batchsize, batchnum):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model_name = []
    simnet = []
    optimizer = []
    values0 = []
    values1 = []
    values2 = []
    test_values0 = []
    test_values1 = []
    test_values2 = []
    for i in range(modelnum):
        values0.append([])
        values1.append([])
        values2.append([])
        test_values0.append([])
        test_values1.append([])
        test_values2.append([])
    loss = nn.MSELoss()
    loss_cla = nn.CrossEntropyLoss()
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters())
    print(rank, world_size)
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    #print(outputs)
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

    # Clean up.
    dist.destroy_process_group()

def main():
    if len(sys.argv) < 3:
        print("Illegal arguments.")
        sys.exit()
    
    batchnum = int(sys.argv[1])
    modelnum = len(sys.argv) - 2
    if batchnum > train_max_num:
        print("Too large training set.")
        sys.exit()
    elif train_max_num % batchnum != 0:
        print("Warning: training set not aligned.")
    stride = math.floor(train_max_num / batchnum)
    
    if large_model:
        batchsize = ori_batch_size // large_model_scale_factor
        batchnum *= large_model_scale_factor
    else:
        batchsize = ori_batch_size

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    f = np.memmap(data_set_name + "/totalall.mmap", dtype=np.float32,
                  mode='r',shape=(total_size, context_length*inst_length))
    fs = np.load(data_set_name + "/statsall.npz")
    print("Train", modelnum, "models with", batchnum*batchsize, ", test with", int(0.5*batchsize))
    world_size = 1
    mp.spawn(rank_train, args=(world_size, f, fs, sys.argv[2:], modelnum, batchsize, batchnum,), nprocs=world_size, join=True)

if __name__=="__main__":
    main()
