import math
import torch
import torch.nn as nn
import torch.nn.functional as F

context_length = 94
#inst_length = 33
inst_length = 39
#inst_length = 43

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
        x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3_F(nn.Module):
    def __init__(self, out, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, ck3, ch3, cs3, cp3, f1):
        super(CNN3_F, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1, stride=cs1, padding=cp1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2, stride=cs2, padding=cp2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3, stride=cs3, padding=cp3)
        self.f1_input = math.floor((context_length + 2 * cp1 - ck1) / cs1 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp2 - ck2) / cs2 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp3 - ck3) / cs3 + 1)
        print(self.f1_input)
        self.f1_input *= ch3
        self.f1_input = int(self.f1_input)
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

class CNN2_F(nn.Module):
    def __init__(self, out, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, f1):
        super(CNN2_F, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1, stride=cs1, padding=cp1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2, stride=cs2, padding=cp2)
        self.f1_input = math.floor((context_length + 2 * cp1 - ck1) / cs1 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp2 - ck2) / cs2 + 1)
        print(self.f1_input)
        self.f1_input *= ch2
        self.f1_input = int(self.f1_input)
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        x = x.view(-1, inst_length, context_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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

class CNN5(nn.Module):
    def __init__(self, out, ck1, ch1, ck2, ch2, ck3, ch3, ck4, ch4, ck5, ch5, f1):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1)
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

class CNN5_F(nn.Module):
    def __init__(self, out, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, ck3, ch3, cs3, cp3, ck4, ch4, cs4, cp4, ck5, ch5, cs5, cp5, f1):
        super(CNN5_F, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inst_length, out_channels=ch1, kernel_size=ck1, stride=cs1, padding=cp1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2, stride=cs2, padding=cp2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3, stride=cs3, padding=cp3)
        self.conv4 = nn.Conv1d(in_channels=ch3, out_channels=ch4, kernel_size=ck4, stride=cs4, padding=cp4)
        self.conv5 = nn.Conv1d(in_channels=ch4, out_channels=ch5, kernel_size=ck5, stride=cs5, padding=cp5)
        self.f1_input = math.floor((context_length + 2 * cp1 - ck1) / cs1 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp2 - ck2) / cs2 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp3 - ck3) / cs3 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp4 - ck4) / cs4 + 1)
        print(self.f1_input)
        self.f1_input = math.floor((self.f1_input + 2 * cp5 - ck5) / cs5 + 1)
        print(self.f1_input)
        self.f1_input *= ch5
        self.f1_input = int(self.f1_input)
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

class CNN3_P2(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3_P2, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length, out_channels=pc, kernel_size=2)
        self.convp2 = nn.Linear(pc, pc)
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
        y = torch.unsqueeze(self.convp2(F.relu(torch.squeeze(y, -1))), -1)
        for i in range(2, context_length):
            xi = torch.cat((x[:, :, 0:1], x[:, :, i:i+1]), 2)
            xo = self.convp(xi)
            xo = torch.unsqueeze(self.convp2(F.relu(torch.squeeze(xo, -1))), -1)
            y = torch.cat((y, xo), 2)
        x = F.relu(y)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3_P_lat(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(CNN3_P_lat, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length, out_channels=pc, kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=pc, out_channels=ch1, kernel_size=ck1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3)
        self.c3_output = ch3 * (context_length - 1 - ck1 - ck2 - ck3 + 3)
        self.f1_input = self.c3_output + context_length * inst_length
        self.fc1 = nn.Linear(self.f1_input, f1)
        self.fc2 = nn.Linear(f1, out)

    def forward(self, x):
        xin = x;
        xin = xin.view(-1, inst_length * context_length)
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
        x = x.view(-1, self.c3_output)
        x = torch.cat((x, xin), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN3_FPB(nn.Module):
    def __init__(self, out, pc, ck1, ch1, cs1, cp1, ck2, ch2, cs2, cp2, ck3, ch3, cs3, cp3, bch1, bch2, bch3, f1):
        super(CNN3_FPB, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length, out_channels=pc, kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=pc, out_channels=ch1, kernel_size=ck1, stride=cs1, padding=cp1)
        self.conv2 = nn.Conv1d(in_channels=ch1, out_channels=ch2, kernel_size=ck2, stride=cs2, padding=cp2)
        self.conv3 = nn.Conv1d(in_channels=ch2, out_channels=ch3, kernel_size=ck3, stride=cs3, padding=cp3)
        self.c3_output = math.floor((context_length -1 + 2 * cp1 - ck1) / cs1 + 1)
        print(self.c3_output)
        self.c3_output = math.floor((self.c3_output + 2 * cp2 - ck2) / cs2 + 1)
        print(self.c3_output)
        self.c3_output = math.floor((self.c3_output + 2 * cp3 - ck3) / cs3 + 1)
        print(self.c3_output)
        self.c3_output *= ch3
        self.c3_output = int(self.c3_output)
        self.bconv1 = nn.Conv1d(in_channels=inst_length, out_channels=bch1, kernel_size=1)
        self.bconv2 = nn.Conv1d(in_channels=bch1, out_channels=bch2, kernel_size=1)
        self.bconv3 = nn.Conv1d(in_channels=bch2, out_channels=bch3, kernel_size=1)
        self.bc3_output = bch3
        self.f1_input = self.c3_output + self.bc3_output
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
        first = x[:, :, 0:1]
        first = F.relu(self.bconv1(first))
        first = F.relu(self.bconv2(first))
        first = F.relu(self.bconv3(first))
        first = first.view(-1, self.bc3_output)
        x = F.relu(y)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.c3_output)
        x = torch.cat((x, first), 1)
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

class FC2_P(nn.Module):
    def __init__(self, out, pc, ck1, ch1, ck2, ch2, ck3, ch3, f1):
        super(FC2_P, self).__init__()
        self.convp = nn.Conv1d(in_channels=inst_length, out_channels=pc, kernel_size=2)
        self.f1_input = pc * (context_length - 1)
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
        x = x.view(-1, self.f1_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
