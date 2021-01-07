import numpy as np
import torch

def get_data(arr, s, low, high, device):
    x = np.copy(arr[low:high,])
    y1 = np.copy(x[:,1:2])
    y2 = np.copy(x[:,3:4])
    y = np.concatenate((y1, y2), axis=1)
    y1 = np.copy(x[:,0:1])
    y1 *= np.sqrt(s['all_var'][0])
    y2 = np.copy(x[:,2:3])
    y2 *= np.sqrt(s['all_var'][2])
    y_cla = np.concatenate((y1, y2), axis=1)
    y_cla = np.rint(y_cla)
    #assert(y_cla.all() >= 0 and y_cla.all() <= 9)
    x[:,0:4] = 0
    x = torch.from_numpy(x.astype('f'))
    y = torch.from_numpy(y.astype('f'))
    y_cla = torch.from_numpy(y_cla.astype(int))
    x = x.to(device)
    y = y.to(device)
    y_cla = y_cla.to(device)
    return x, y, y_cla

def train_com(net, x, y, y_cla, loss0, loss1, loss2, loss, loss_cla, optimizer):
    output = net(x)
    #output, fc, rc = net(x)
    value0 = loss(output[:,0:2],y)
    loss0 += value0.cpu().item()
    value1 = loss_cla(output[:,2:12],y_cla[:,0:1].view(-1))
    loss1 += value1.cpu().item()
    value2 = loss_cla(output[:,12:22],y_cla[:,1:2].view(-1))
    loss2 += value2.cpu().item()
    optimizer.zero_grad()
    all_loss = 20 * value0 + value1 + value2
    all_loss.backward()
    optimizer.step()
    return loss0, loss1, loss2

def test_com(net, x, y, y_cla, loss, loss_cla):
    output = net(x)
    #output, fc, rc = net(x)
    value0 = loss(output[:,0:2],y)
    value1 = loss_cla(output[:,2:12],y_cla[:,0:1].view(-1))
    value2 = loss_cla(output[:,12:22],y_cla[:,1:2].view(-1))
    return value0.cpu().item(), value1.cpu().item(), value2.cpu().item()

def print_arr(arr):
    print(', '.join('{:0.5f}'.format(i) for i in arr))

def generate_model_name(name):
    name = name.replace(" ", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    name = name.replace("'", "_")
    name = name.replace("\"", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    return name
