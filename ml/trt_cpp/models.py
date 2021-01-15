import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import context_length, inst_length
#from enet import Efficient1DNet

class Fusion1dFC(nn.Module):
    def __init__(self, out):
        super(Fusion1d, self).__init__()
        self.out = out
        self._fu0 = nn.Linear(inst_length, out, bias=False)
        #self._fu1 = nn.Linear(inst_length, out, bias=False)
        self._conv = nn.Conv1d(in_channels=inst_length,
                               out_channels=out,
                               kernel_size=1, bias=False)

    def forward(self, x):
        x = x.view(-1, context_length, inst_length)
        x0 = self._fu0(x[:, 0, :])
        x0 = x0.view(-1, self.out, 1)
        x0 = x0.expand(-1, -1, context_length - 1)
        x = x.view(-1, context_length, inst_length).transpose(2,1)
        x = self._conv(x[:, :, 1:context_length])
        x += x0
        #y = x.new(x.size(0), self.out, context_length - 1)
        ##print(y.size())
        #for i in range(1, context_length):
        #    y[:, :, i-1] = self._fu1(x[:, i, :]) + x0
        #x = F.relu(y)
        x = F.relu(x)
        return x

class Fusion1d(nn.Module):
    def __init__(self, out):
        super(Fusion1d2, self).__init__()
        self._conv = nn.Conv1d(in_channels=inst_length*2, out_channels=out, kernel_size=1, bias=False)

    def forward(self, x):
        #x = x.view(-1, inst_length, context_length)
        x = x.view(-1, context_length, inst_length).transpose(2,1)
        #copies = x[:,:,0].unsqueeze(-1)
        copies = x[:,:,0:1]
        #copies = copies.repeat(1, 1, context_length - 1)
        copies = copies.expand(-1, -1, context_length - 1)
        #test = torch.cat((x,copies),-2)
        x = torch.cat((x[:,:,1:context_length],copies),1)
        #print(x.size())
        x = self._conv(x)
        x = F.relu(x)
        return x

class FC2(nn.Module):
    def __init__(self, out, f1):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(inst_length*context_length, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        x = x.view(-1, inst_length*context_length)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3(nn.Module):
    def __init__(self, out, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.f1_input = ch3 * (context_length - ck1 - ck2 - ck3 + 3)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        #x = x.view(-1, inst_length, context_length)
        x = x.view(-1, context_length, inst_length).transpose(2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


