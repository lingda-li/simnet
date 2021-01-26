from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from custom_data import *
from utils import profile_model, generate_model_name
from models import *
from cfg_qq import *


lat_loss_fn = nn.MSELoss()
cla_loss_fn = nn.CrossEntropyLoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_lat_loss = 0
    total_cla_loss1 = 0
    total_cla_loss2 = 0
    total_cla_loss3 = 0
    start_t = time.time()
    print_threshold = len(train_loader) // 100
    for batch_idx, (data, lat_target, cla_target) in enumerate(train_loader):
        data, lat_target, cla_target = data.to(device), lat_target.to(device), cla_target.to(device)
        optimizer.zero_grad()
        output = model(data)
        lat_loss = lat_loss_fn(output[:,0:3], lat_target)
        cla_loss1 = cla_loss_fn(output[:,3:3+num_classes], cla_target[:,0])
        cla_loss2 = cla_loss_fn(output[:,3+num_classes:3+2*num_classes], cla_target[:,1])
        cla_loss3 = cla_loss_fn(output[:,3+2*num_classes:3+3*num_classes], cla_target[:,2])
        loss = 0.05 * lat_loss + cla_loss1 + cla_loss2 + cla_loss3
        total_lat_loss += lat_loss.item()
        total_cla_loss1 += cla_loss1.item()
        total_cla_loss2 += cla_loss2.item()
        total_cla_loss3 += cla_loss3.item()
        loss.backward()
        optimizer.step()
        if batch_idx % print_threshold == print_threshold - 1:
            print('.', flush=True, end='')
        if args.dry_run:
            break
    total_lat_loss /= len(train_loader.dataset) / 65536
    total_cla_loss1 /= len(train_loader.dataset) / 65536
    total_cla_loss2 /= len(train_loader.dataset) / 65536
    total_cla_loss3 /= len(train_loader.dataset) / 65536
    print('\nTrain Epoch: {} \tLat Loss: {:.6f} \tCla Loss1: {:.6f} \tCla Loss2: {:.6f} \tCla Loss3: {:.6f} \tTime: {:.1f}'.format(
        epoch, total_lat_loss, total_cla_loss1, total_cla_loss2, total_cla_loss3, time.time() - start_t))


def test(model, device, test_loader):
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
    total_lat_loss /= len(test_loader.dataset) / 65536
    total_cla_loss1 /= len(test_loader.dataset) / 65536
    total_cla_loss2 /= len(test_loader.dataset) / 65536
    total_cla_loss3 /= len(test_loader.dataset) / 65536
    print('Test set: Lat Loss: {:.6f} \tCla Loss1: {:.6f} \tCla Loss2: {:.6f} \tCla Loss3: {:.6f}'.format(
        total_lat_loss, total_cla_loss1, total_cla_loss2, total_cla_loss3))


def save_checkpoint(name, model, optimizer, epoch):
    name = 'checkpoints/' + generate_model_name(name, epoch) + '.pt'
    saved_dict = {'epoch': epoch,
                  'optimizer_state_dict': optimizer.state_dict()}
    if torch.cuda.device_count() > 1:
        model_dict = {'model_state_dict': model.module.state_dict()}
    else:
        model_dict = {'model_state_dict': model.state_dict()}
    saved_dict.update(model_dict)
    torch.save(saved_dict, name)
    print("Saved checkpoint", name)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SIMNET Training')
    parser.add_argument('--batch-size', type=int, default=1024*64, metavar='N',
                        help='input batch size (default: 1024*64)')
    parser.add_argument('--train-size', type=int, default=1024*64, metavar='N',
                        help='input size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many epochs to save models')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': False}
        kwargs.update(cuda_kwargs)

    dataset1 = QQDataset(data_file_name, total_size, context_length*inst_length, 0, args.train_size, num_classes=num_classes)
    dataset2 = QQDataset(data_file_name, total_size, context_length*inst_length, test_start, test_end, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    assert len(args.models) == 1
    model = eval(args.models[0])
    profile_model(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    #scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()
        if args.save_model and epoch % args.save_interval == 0:
            save_checkpoint(args.models[0], model, optimizer, epoch)


if __name__ == '__main__':
    main()
