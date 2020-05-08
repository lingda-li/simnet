import torch
import torch.nn as nn
import torch.nn.functional as F

context_length = 94
inst_length = 39

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
        x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2_P(nn.Module):
    def __init__(self, out, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN2_P, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.f1_input = ch3 * (context_length - 1 - ck2 - ck3 + 2)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)
        xi = torch.cat((x[:, :, 0:1], x[:, :, 1:2]), 2)
        y = self.conv1(xi)
        for i in range(2, context_length):
            xi = torch.cat((x[:, :, 0:1], x[:, :, i:i+1]), 2)
            xo = self.conv1(xi)
            y = torch.cat((y, xo), 2)
        x = F.relu(y)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3_P(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3_P, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length, out_channels=pc, kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=pc, out_channels=ch1, kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.f1_input = ch3 * (context_length - 1 - ck1 - ck2 - ck3 + 3)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)
        xi = torch.cat((x[:, :, 0:1], x[:, :, 1:2]), 2)
        y = self.convp(xi)
        for i in range(2, context_length):
            xi = torch.cat((x[:, :, 0:1], x[:, :, i:i+1]), 2)
            xo = self.convp(xi)
            y = torch.cat((y, xo), 2)
        x = F.relu(y)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3_P_D(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3_P_D, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length,    # e.g. 33
                               out_channels=pc,  # e.g. 64
                               kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=pc, out_channels=ch1, kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.f1_input = ch3 * (context_length - 1 - ck1 - ck2 - ck3 + 3)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

        self.newlin = nn.Linear(pc,pc)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)

        xi = torch.cat((x[:, :, 0:1], x[:, :, 1:2]), 2)
        y = self.convp(xi) # x 1

        y = torch.squeeze(y,-1)
        y = F.relu(y)
        y = self.newlin(y) 
        y = torch.unsqueeze(y,-1) 

        for i in range(2, context_length):
            xi = torch.cat((x[:, :, 0:1], x[:, :, i:i+1]), 2)
            xo = self.convp(xi) # batch x 33 x 2 -> batchsize x 64 x 1
            
            xo = torch.squeeze(xo,-1) # batch x 64
            xo = F.relu(xo)
            xo = self.newlin(xo) 
            xo = torch.unsqueeze(xo,-1) # batch x 64  x 1

            y = torch.cat((y, xo), 2)

        x = F.relu(y)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN3_P_D_C(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3_P_D_C, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length,    # e.g. 33
                               out_channels=pc,  # e.g. 64
                               kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=pc, out_channels=ch1, kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.f1_input = ch3 * (context_length  - ck1 - ck2 - ck3 + 3)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

        self.convt = nn.Conv1d(in_channels=inst_length*2,
                               out_channels=pc,
                               kernel_size=1)

        self.convd = nn.Conv1d(in_channels = pc,
                               out_channels = pc,
                               kernel_size = 1)

        self.newlin = nn.Linear(pc,pc)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)

        copies = x[:,:,0].unsqueeze(-1)
        copies=copies.repeat(1,1, context_length)
        test  = torch.cat((x,copies),-2)

        y = self.convt(test)
        y = F.tanh(y) # F.relu(y)
        y = self.convd(y)
        y = F.tanh(y) #F.relu(y)

        x = F.relu(self.conv1(y))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN5(nn.Module):
    def __init__(self, out, ck1, ch1, ck2, ch2, ck3, ch3, ck4, ch4, ck5, ch5, f1):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length,  # e.g. 33
                               out_channels=ch1, 
                               kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.conv4 = nn.Conv1d(in_channels=ch3, out_channels=ch4, kernel_size=ck4)
        self.conv5 = nn.Conv1d(in_channels=ch4, out_channels=ch5, kernel_size=ck5)
        self.f1_input = ch5 * (context_length - ck1 - ck2 - ck3 - ck4 - ck5 + 5)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(MLP, self).__init__()

        self.convt = nn.Conv1d(in_channels=inst_length*2,
                               out_channels=pc,
                               kernel_size=1)

        self.convd = nn.Conv1d(in_channels = pc,
                               out_channels = pc,
                               kernel_size = 1)

        self.f1_input = context_length*pc
        self.fc1 = nn.Linear(self.f1_input, 100)
        self.fc2 = nn.Linear(100, out)

        self.newlin = nn.Linear(pc,pc)
    
    def forward(self, x):
    
        x = x.view(-1, inst_length, context_length)
        
        copies = x[:,:,0].unsqueeze(-1)
        copies=copies.repeat(1,1, context_length)
        test  = torch.cat((x,copies),-2)
        y = self.convt(test)
        y = F.tanh(y)
        y = self.convd(y)
        y = F.tanh(y)
        #print(y.shape)
        #print(self.f1_input)
        x = y.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
