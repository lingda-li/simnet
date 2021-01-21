import numpy as np
import torch

# Detailed parameters.
num_classes = 100


def get_cla_data(arr, s, low, high, device):
    x = np.copy(arr[low:high,])
    y1 = np.copy(x[:,1:2])
    y1 *= np.sqrt(s['all_var'][1])
    y = np.rint(y1)
    y[y > num_classes - 1] = num_classes - 1
    x[:,0:4] = 0
    x = torch.from_numpy(x.astype('f'))
    y = torch.from_numpy(y.astype(int))
    x = x.to(device)
    y = y.to(device)
    return x, y


def train_cla(net, x, y, total_loss, loss_func, optimizer):
    output = net(x)
    value = loss_func(output,y.view(-1))
    total_loss += value.cpu().item()
    optimizer.zero_grad()
    value.backward()
    optimizer.step()
    return total_loss


def test_cla(net, x, y, loss_func):
    output = net(x)
    value = loss_func(output,y.view(-1))
    return value.cpu().item()
