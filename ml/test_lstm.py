import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_data import *
#from utils import get_inst_type
from models_lstm import *
#from cfg_lstm import *
from cfg_lstm_fs import *


test_seq_length = 100000000
#test_seq_length = 100000
lat_loss_fn = nn.MSELoss()
cla_loss_fn = nn.CrossEntropyLoss()


def analyze_lat(args, output, target):
    print("\toutput:", output)
    print("\ttarget:", target)
    int_output = np.rint(output)
    print("\tnorm output:", int_output)
    target = np.rint(target)
    errs = target - int_output
    print("\terrors:", errs)
    errs = errs.ravel()
    #errs[errs < 0] = -errs[errs < 0]

    flat_target = target.ravel()
    print("\tErr avg, persentage, and std:", np.average(np.abs(errs)), "\t", np.sum(errs) / np.sum(flat_target), "\t", np.std(errs))
    his = np.histogram(errs, bins=range(-1, 100))
    print("\t", his[0] / errs.size)
    errs /= flat_target + 1
    print("\tNorm err avg and std:", np.average(np.abs(errs)), "\t", np.std(errs))

def analyze(args, output, lat_target, cla_target=None, cla_output=None):
    print("Size:", output.shape)
    lat_target = lat_target.detach().numpy()
    if args.no_class:
        assert cla_target is None
    else:
        assert cla_target is not None
        cla_target = cla_target.detach().numpy()
    for i in range(tgt_length):
        print(i, ":")
        lat_output = output.detach().numpy()
        lat_output = lat_output[:,i]
        cur_lat_target = lat_target[:,i]

        if args.no_class:
            cur_output = lat_output
        else:
            cla_idx = -1
            if i < 2:
                cla_idx = i
            elif i == tgt_length - 1:
                cla_idx = 2
            if cla_idx != -1:
                if cla_output is not None:
                    cla_res = torch.argmax(cla_output[:,num_classes*cla_idx:num_classes*(cla_idx+1)], dim=1)
                else:
                    cla_res = torch.argmax(output[:,num_classes*cla_idx+tgt_length:num_classes*(cla_idx+1)+tgt_length], dim=1)
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


def analyze_seq(args, output, lat_target, cla_target=None, cla_output=None):
    print("Size:", output.shape)
    lat_target = lat_target.detach().numpy()
    if args.no_class:
        assert cla_target is None
    else:
        assert cla_target is not None
        cla_target = cla_target.detach().numpy()
    for i in range(1):
        print(i, ":")
        lat_output = output.detach().numpy()
        lat_output = lat_output[:,i]
        cur_lat_target = lat_target[:,i]

        if args.no_class:
            cur_output = lat_output
        else:
            cla_idx = -1
            if i < 2:
                cla_idx = i
            elif i == tgt_length - 1:
                cla_idx = 2
            if cla_idx != -1:
                if cla_output is not None:
                    cla_res = torch.argmax(cla_output[:,num_classes*cla_idx:num_classes*(cla_idx+1)], dim=1)
                else:
                    cla_res = torch.argmax(output[:,num_classes*cla_idx+tgt_length:num_classes*(cla_idx+1)+tgt_length], dim=1)
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

        print("\toutput:", cur_output)
        print("\ttarget:", cur_lat_target)
        int_output = np.rint(cur_output)
        print("\tnorm output:", int_output)
        target = np.rint(cur_lat_target)
        errs = target - int_output
        print("\terrors:", errs)
        errs = errs.ravel()
        #errs[errs < 0] = -errs[errs < 0]

        flat_target = target.ravel()
        print("\tErr avg, persentage, and std:", np.average(np.abs(errs)), "\t", np.sum(errs) / np.sum(flat_target), "\t", np.std(errs))
        his = np.histogram(errs, bins=range(-1, 100))
        print("\t", his[0] / errs.size)
        errs /= flat_target + 1
        print("\tNorm err avg and std:", np.average(np.abs(errs)), "\t", np.std(errs))

        target = target.reshape((len(sim_datasets), test_seq_length))
        int_output = int_output.reshape((len(sim_datasets), test_seq_length))
        sim_targets = np.sum(target, axis=1)
        sim_results = np.sum(int_output, axis=1)
        np.set_printoptions(threshold=np.inf)
        print("Simulation targets:", sim_targets)
        print("Simulation results:", sim_results)
        sim_errs = (sim_results - sim_targets) / sim_targets
        print("Simulation errors:", sim_errs)
        print("Avg error:", np.abs(sim_errs).mean())
        #com_targets = np.copy(target)
        #com_outputs = np.copy(int_output)
        #for j in range(1, test_seq_length):
        #    for k in range(len(sim_datasets)):
        #        com_targets[k, j] += com_targets[k, j-1]
        #        com_outputs[k, j] += com_outputs[k, j-1]
        #print("\tcom_targets:", com_targets)
        #print("\tcom_outputs:", com_outputs)
        #errs = com_targets - com_outputs
        #errs /= com_targets
        #print("\tcom_errors:", errs)
        #errs = np.mean(np.abs(errs), axis=0)
        #print("\tcom_errors avg:", errs)
        #errs = errs.reshape((-1,1024))
        #np.set_printoptions(threshold=np.inf)
        #print("\tcom_errors chunk avg:", errs.mean(axis=1))
            

def simulate(args, model, device, test_loaders):
    model.eval()
    total_lat_loss = 0
    total_cla_loss1 = 0
    total_cla_loss2 = 0
    total_cla_loss3 = 0
    with torch.no_grad():
        if args.no_class:
            total_output = torch.zeros(0, tgt_length)
            total_lat_target = torch.zeros(0, tgt_length)
            for i in range(len(test_loaders)):
                for data, lat_target in test_loaders[i]:
                    total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                    data, lat_target = data.to(device), lat_target.to(device)
                    output = model(data)
                    total_lat_loss += lat_loss_fn(output[:,0:tgt_length], lat_target).item()
                    if not args.no_cuda:
                        output = output.cpu()
                    total_output = torch.cat((total_output, output), 0)
        else:
            total_output = torch.zeros(0, tgt_length+3*num_classes)
            total_lat_target = torch.zeros(0, tgt_length)
            total_cla_target = torch.zeros(0, 3)
            for i in range(len(test_loaders)):
                for data, lat_target, cla_target in test_loaders[i]:
                    total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                    total_cla_target = torch.cat((total_cla_target, cla_target), 0)
                    data, lat_target, cla_target = data.to(device), lat_target.to(device), cla_target.to(device)
                    output = model(data)
                    total_lat_loss += lat_loss_fn(output[:,0:tgt_length], lat_target).item()
                    total_cla_loss1 += cla_loss_fn(output[:,tgt_length:tgt_length+num_classes], cla_target[:,0]).item()
                    total_cla_loss2 += cla_loss_fn(output[:,tgt_length+num_classes:tgt_length+2*num_classes], cla_target[:,1]).item()
                    total_cla_loss3 += cla_loss_fn(output[:,tgt_length+2*num_classes:tgt_length+3*num_classes], cla_target[:,2]).item()
                    if not args.no_cuda:
                        output = output.cpu()
                    total_output = torch.cat((total_output, output), 0)
    total_lat_loss /= len(test_loaders[0]) * len(test_loaders)
    if args.no_class:
        print('Test set: Lat Loss: {:.6f}'.format(total_lat_loss), flush=True)
        analyze_seq(args, total_output, total_lat_target)
    else:
        total_cla_loss1 /= len(test_loaders[0]) * len(test_loaders)
        total_cla_loss2 /= len(test_loaders[0]) * len(test_loaders)
        total_cla_loss3 /= len(test_loaders[0]) * len(test_loaders)
        print('Test set: Lat Loss: {:.6f} \tCla Loss1: {:.6f} \tCla Loss2: {:.6f} \tCla Loss3: {:.6f}'.format(
            total_lat_loss, total_cla_loss1, total_cla_loss2, total_cla_loss3), flush=True)
        analyze_seq(args, total_output, total_lat_target, total_cla_target)


def test(args, model, device, test_loader):
    model.eval()
    total_lat_loss = 0
    total_cla_loss1 = 0
    total_cla_loss2 = 0
    total_cla_loss3 = 0
    with torch.no_grad():
        if args.no_class:
            total_output = torch.zeros(0, tgt_length)
            total_lat_target = torch.zeros(0, tgt_length)
            for data, lat_target in test_loader:
                total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                data, lat_target = data.to(device), lat_target.to(device)
                output = model(data)
                total_lat_loss += lat_loss_fn(output[:,0:tgt_length], lat_target).item()
                if not args.no_cuda:
                    output = output.cpu()
                total_output = torch.cat((total_output, output), 0)
        else:
            total_output = torch.zeros(0, tgt_length+3*num_classes)
            total_lat_target = torch.zeros(0, tgt_length)
            total_cla_target = torch.zeros(0, 3)
            for data, lat_target, cla_target in test_loader:
                total_lat_target = torch.cat((total_lat_target, lat_target), 0)
                total_cla_target = torch.cat((total_cla_target, cla_target), 0)
                data, lat_target, cla_target = data.to(device), lat_target.to(device), cla_target.to(device)
                output = model(data)
                total_lat_loss += lat_loss_fn(output[:,0:tgt_length], lat_target).item()
                total_cla_loss1 += cla_loss_fn(output[:,tgt_length:tgt_length+num_classes], cla_target[:,0]).item()
                total_cla_loss2 += cla_loss_fn(output[:,tgt_length+num_classes:tgt_length+2*num_classes], cla_target[:,1]).item()
                total_cla_loss3 += cla_loss_fn(output[:,tgt_length+2*num_classes:tgt_length+3*num_classes], cla_target[:,2]).item()
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
    traced_script_module = torch.jit.trace(model, torch.rand(1, seq_length, input_length).to(device))
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

    dataset = ComSeqDataset(4, test_start, test_end)
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
    test_loaders = []
    for i in range(len(sim_datasets)):
        seq_dataset = SeqDataset(sim_datasets[i][0], sim_datasets[i][1], 0, test_seq_length)
        test_loaders.append(torch.utils.data.DataLoader(seq_dataset, **kwargs))
        simulate(args, model, device, test_loaders)
    if not args.no_save:
        save_ts_model(args.checkpoints, model, device)


if __name__ == '__main__':
    main()
