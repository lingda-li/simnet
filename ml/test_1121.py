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
from cfg import *


lat_loss_fn = nn.MSELoss()
cla_loss_fn = nn.CrossEntropyLoss()
inst_type = -2


def analyze_lat(args, output, target):
    print("\toutput:", output)
    print("\ttarget:", target)
    int_output = np.rint(output)
    print("\tnorm output:", int_output)
    target = np.rint(target)
    errs = target - int_output
    print("\terrors:", errs)
    errs = errs.ravel()
    errs[errs < 0] = -errs[errs < 0]

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

    flat_target = target.ravel()
    print("\tErr avg, persentage, and std:", np.average(errs[errs != -1]), "\t", np.sum(errs[errs != -1]) / np.sum(flat_target[errs != -1]), "\t", np.std(errs[errs != -1]))
    errs /= flat_target + 1
    print("\tNorm err avg, persentage, and std:", np.average(errs[errs != -1]), "\t", np.sum(errs[errs != -1]) / np.sum(flat_target[errs != -1]), "\t", np.std(errs[errs != -1]))
    his = np.histogram(errs, bins=range(-1, 100))
    print("\tdata percentage:", errs[errs != -1].size / errs.size)
    print("\t", his[0] / errs[errs != -1].size)

def analyze(args, output, lat_target, cla_target=None, cla_output=None):
    print("Size:", output.shape)
    lat_target = lat_target.detach().numpy()
    if args.no_class:
        assert cla_target is None
    else:
        assert cla_target is not None
        cla_target = cla_target.detach().numpy()
    complete_sum = np.zeros(lat_target.shape[0])
    complete_target_sum = np.zeros(lat_target.shape[0])
    store_sum = np.zeros(lat_target.shape[0])
    store_target_sum = np.zeros(lat_target.shape[0])
    cal_sum = False
    cal_st_sum = False
    if target_length == input_start:
        cal_sum = True
        cal_st_sum = True
        component_start = 3
        component_end = input_start - 3
        st_component_end = input_start - 1
    elif target_length == input_start - 2:
        cal_sum = True
        cal_st_sum = True
        component_start = 1
        component_end = input_start - 5
        st_component_end = input_start - 3
    elif target_length == input_start - 3:
        cal_sum = True
        component_start = 2
        component_end = input_start - 4
    for i in range(target_length):
        print(i, ":")
        lat_output = output.detach().numpy()
        lat_output = lat_output[:,i]
        cur_lat_target = lat_target[:,i]
        if cal_sum and i >= component_start and i < component_end:
            complete_sum += lat_output
            complete_target_sum += cur_lat_target
        if cal_st_sum and i >= component_start and i < st_component_end:
            store_sum += lat_output
            store_target_sum += cur_lat_target

        if args.no_class:
            cur_output = lat_output
        else:
            cla_idx = -1
            if i < 2:
                cla_idx = i
            elif i == target_length - 1:
                cla_idx = 2
            if cla_idx != -1:
                if cla_output is not None:
                    cla_res = torch.argmax(cla_output[:,num_classes*cla_idx:num_classes*(cla_idx+1)], dim=1)
                else:
                    cla_res = torch.argmax(output[:,num_classes*cla_idx+target_length:num_classes*(cla_idx+1)+target_length], dim=1)
                cla_res = cla_res.detach().numpy()
                cur_cla_target = cla_target[:,cla_idx]
                print("\tclass output:", cla_res)
                print("\tclass target:", cur_cla_target)
                if cla_idx == 0: # Fetch latency.
                    com_output = np.where(cla_res < num_classes - 1, cla_res, lat_output)
                elif cla_idx == 1: # Completion latency.
                    com_output = np.where(cla_res < num_classes - 1, cla_res + min_complete_lat, lat_output)
                else:
                    #com_output = np.where(cla_res < num_classes - 1, cla_res + (min_store_lat - 1), lat_output)
                    #com_output = np.where(cla_res == 0, 0, com_output)
                    com_output = np.where(cla_res < num_classes - 1, cla_res, lat_output)
                print("\tcombined output:", com_output)
                cur_output = com_output
            else:
                cur_output = lat_output

        analyze_lat(args, cur_output, cur_lat_target)

    if cal_sum:
        print("Combined complete:")
        analyze_lat(args, complete_sum, complete_target_sum)
    if cal_st_sum:
        print("Combined store:")
        analyze_lat(args, store_sum, store_target_sum)


def test(args, model, device, test_loader):
    model.eval()
    total_lat_loss = 0
    total_cla_loss1 = 0
    total_cla_loss2 = 0
    total_cla_loss3 = 0
    with torch.no_grad():
        if args.no_class:
            total_output = torch.zeros(0, target_length)
            total_lat_target = torch.zeros(0, target_length)
            for data, lat_target in test_loader:
                total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                data, lat_target = data.to(device), lat_target.to(device)
                output = model(data)
                total_lat_loss += lat_loss_fn(output[:,0:target_length], lat_target).item()
                if not args.no_cuda:
                    output = output.cpu()
                total_output = torch.cat((total_output, output), 0)
        else:
            total_output = torch.zeros(0, target_length+3*num_classes)
            total_lat_target = torch.zeros(0, target_length)
            total_cla_target = torch.zeros(0, 3)
            for data, lat_target, cla_target in test_loader:
                total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                total_cla_target = torch.cat((total_cla_target, cla_target), 0)
                data, lat_target, cla_target = data.to(device), lat_target.to(device), cla_target.to(device)
                output = model(data)
                total_lat_loss += lat_loss_fn(output[:,0:target_length], lat_target).item()
                total_cla_loss1 += cla_loss_fn(output[:,target_length:target_length+num_classes], cla_target[:,0]).item()
                total_cla_loss2 += cla_loss_fn(output[:,target_length+num_classes:target_length+2*num_classes], cla_target[:,1]).item()
                total_cla_loss3 += cla_loss_fn(output[:,target_length+2*num_classes:target_length+3*num_classes], cla_target[:,2]).item()
                if not args.no_cuda:
                    output = output.cpu()
                total_output = torch.cat((total_output, output), 0)
    total_lat_loss /= len(test_loader)
    if args.no_class:
        print('Test set: Lat Loss: {:.6f}'.format(total_lat_loss), flush=True)
        analyze(args, total_output, total_lat_target)
    else:
        total_cla_loss1 /= len(test_loader)
        total_cla_loss2 /= len(test_loader)
        total_cla_loss3 /= len(test_loader)
        print('Test set: Lat Loss: {:.6f} \tCla Loss1: {:.6f} \tCla Loss2: {:.6f} \tCla Loss3: {:.6f}'.format(
            total_lat_loss, total_cla_loss1, total_cla_loss2, total_cla_loss3), flush=True)
        analyze(args, total_output, total_lat_target, total_cla_target)


def load_checkpoint(name, model, training=False, optimizer=None):
    assert 'checkpoints/' in name
    cp = torch.load(name, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model_state_dict'])
    if training:
        assert optimizer is not None
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    print("Loaded checkpoint", name)
    print("epoch:", cp['epoch'])


def save_ts_model(name, model, device):
    assert 'checkpoints/' in name
    name = name.replace('checkpoints/', 'models/')
    model.eval()
    traced_script_module = torch.jit.trace(model, torch.rand(1, context_length * input_length).to(device))
    traced_script_module.save(name)
    print("Saved model", name)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SIMNET Testing')
    parser.add_argument('--batch-size', type=int, default=1024*64, metavar='N',
                        help='input batch size (default: 1024*64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-class', action='store_true', default=False,
                        help='disables classification')
    parser.add_argument('--no-save', action='store_true', default=False,
                        help='do not save model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    dataset = CompressedDataset(data_file_name, total_size + 1, total_inst_num, test_start, test_end, num_classes=num_classes)
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
    if not args.no_save:
        save_ts_model(args.checkpoints, model, device)


if __name__ == '__main__':
    main()
