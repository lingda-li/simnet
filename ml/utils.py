import numpy as np
import torch
from cfg import context_length, inst_length, is_save_model


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


def generate_model_name(name, epochs=None):
    name = name.replace(" ", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    name = name.replace("'", "_")
    name = name.replace("\"", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    if epochs is not None:
        name += "_e" + str(epochs)
    return name


def save_model(model, name, device):
    if not is_save_model:
        return
    if device.type == 'cuda':
        torch.save(model.module, 'models/' + name)
    else:
        torch.save(model, 'models/' + name)
    print("Saved model", name)


def get_inst(vals, n, fs=None, use_mean=False):
    inst = vals[inst_length*n:inst_length*(n+1)]
    if fs is not None:
        inst *= np.sqrt(fs['all_var'])
        if use_mean:
            inst += fs['all_mean']
    else:
        assert(not use_mean)
    return inst


def get_inst_field(vals, n, i, fs=None, use_mean=False):
    inst_field = vals[inst_length * n + i]
    if fs is not None:
        inst_field *= np.sqrt(fs['all_var'][i])
        if use_mean:
            inst_field += fs['all_mean'][i]
    else:
        assert(not use_mean)
    inst_field = np.rint(inst_field)
    return inst_field


def get_inst_type(vals, n, fs=None, use_mean=False):
    return get_inst_field(vals, n, 4, fs, use_mean)
