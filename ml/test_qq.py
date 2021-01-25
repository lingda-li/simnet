import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_data import *
from utils import profile_model, get_inst_type
from models import *
from cfg_qq import *


lat_loss_fn = nn.MSELoss()
cla_loss_fn = nn.CrossEntropyLoss()
inst_type = -2


def analyze(args, output, lat_target, cla_target, data, cla_output=None):
    lat_target = lat_target.detach().numpy()
    cla_target = cla_target.detach().numpy()
    data = data.detach().numpy()
    for i in range(3):
        lat_output = output.detach().numpy()
        lat_output = lat_output[:,i]
        cur_lat_target = lat_target[:,i]
        print("latency output:", lat_output)
        print("latency target:", cur_lat_target)
        lat_output = np.rint(lat_output)
        cur_lat_target = np.rint(cur_lat_target)
        print("norm latency output:", lat_output)
    
        if args.no_class:
            errs = cur_lat_target - lat_output
        else:
            if cla_output is not None:
                cla_res = torch.argmax(cla_output[:,num_classes*i:num_classes*(i+1)], dim=1)
            else:
                cla_res = torch.argmax(output[:,num_classes*i+3:num_classes*(i+1)+3], dim=1)
            cla_res = cla_res.detach().numpy()
            cur_cla_target = cla_target[:,i]
            print("class output:", cla_res)
            print("class target:", cur_cla_target)
            if i == 0: # Fetch latency.
                com_output = np.where(cla_res < num_classes - 1, cla_res, lat_output)
            elif i == 1: # Completion latency.
                com_output = np.where(cla_res < num_classes - 1, cla_res + 6, lat_output)
            else:
                com_output = np.where(cla_res < num_classes - 1, cla_res + 9, lat_output)
                com_output = np.where(cla_res == 0, 0, com_output)
            print("combined output:", com_output)
            errs = cur_lat_target - com_output
    
        print("errors:", errs)
        errs = errs.ravel()
        errs[errs < 0] = -errs[errs < 0]
        #errs[cur_cla_target == num_classes - 1] = -1
        print(errs.size)
    
        if inst_type >= -1:
            for i in range(errs.size):
                cur_inst_type = get_inst_type(x[i], 0, fs) - 1
                #print(cur_inst_type)
                assert cur_inst_type >= 0 and cur_inst_type < 37
                if inst_type >= 0 and cur_inst_type != inst_type:
                    errs[i] = -1
                elif inst_type == -1 and (cur_inst_type == 25 or cur_inst_type == 26):
                    errs[i] = -1
            print(errs)
    
        flat_target = cur_lat_target.ravel()
        print("Err avg, persentage, and std:", np.average(errs[errs != -1]), "\t", np.sum(errs[errs != -1]) / np.sum(flat_target[errs != -1]), "\t", np.std(errs[errs != -1]))
        his = np.histogram(errs, bins=range(-1, 100))
        print("data percentage:", errs[errs != -1].size / errs.size)
        print(his[0] / errs[errs != -1].size)


def test(args, model, device, test_loader):
    model.eval()
    total_lat_loss = 0
    total_cla_loss1 = 0
    total_cla_loss2 = 0
    total_cla_loss3 = 0
    with torch.no_grad():
        for data, lat_target, cla_target in test_loader:
            data, lat_target, cla_target = data.to(device), lat_target.to(device), cla_target.to(device)
            output = model(data)
            total_lat_loss += lat_loss_fn(output[:,0:3], lat_target).item()
            total_cla_loss1 += cla_loss_fn(output[:,3:3+num_classes], cla_target[:,0]).item()
            total_cla_loss2 += cla_loss_fn(output[:,3+num_classes:3+2*num_classes], cla_target[:,1]).item()
            total_cla_loss3 += cla_loss_fn(output[:,3+2*num_classes:3+3*num_classes], cla_target[:,2]).item()
            analyze(args, output, lat_target, cla_target, data)
    total_lat_loss /= len(test_loader) * test_loader.batch_size / 65536
    total_cla_loss1 /= len(test_loader) * test_loader.batch_size / 65536
    total_cla_loss2 /= len(test_loader) * test_loader.batch_size / 65536
    total_cla_loss3 /= len(test_loader) * test_loader.batch_size / 65536
    print('Test set: Lat Loss: {:.6f} \tCla Loss1: {:.6f} \tCla Loss2: {:.6f} \tCla Loss3: {:.6f}'.format(
        total_lat_loss, total_cla_loss1, total_cla_loss2, total_cla_loss3), flush=True)


def load_checkpoint(name, model, training=False, optimizer=None):
    assert 'checkpoints/' in name
    cp = torch.load(name)
    model.load_state_dict(cp['model_state_dict'])
    if training:
        assert optimizer is not None
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    print("Loaded checkpoint", name)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SIMNET Testing')
    parser.add_argument('--batch-size', type=int, default=1024*64, metavar='N',
                        help='input batch size (default: 1024*64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-class', action='store_true', default=False,
                        help='disables classification ')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    dataset = QQDataset(data_file_name, total_size, context_length*inst_length, test_start, test_end, num_classes=num_classes)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        kwargs.update(cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    assert len(args.models) == 1
    model = eval(args.models[0])
    load_checkpoint(args.checkpoints, model)
    #profile_model(model)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
