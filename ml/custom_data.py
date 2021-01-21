import torch
import numpy as np
from torch.utils.data import Dataset


class MemoryMappedDataset(Dataset):

    def __init__(self, mmapped_arr):
        self.mmapped_arr = mmapped_arr

    def __len__(self):
        return len(self.mmapped_arr)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = np.copy(self.mmapped_arr[idx])

        y = np.copy(x[1:2]) # x[0] is fetch classification data, x[1] is fetch latency
        y2 = np.copy(x[3:4]) # x[2] is completion classification x[3] is completion latency
        y = np.concatenate((y, y2), axis=0) #target is the fetch and completion time
        x[0:4] = 0 #setting the target data to zero now.

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x,y


class ComDataset(Dataset):

    def __init__(self, file_name, rows, columns, stat_file_name=None):
        self.arr = np.memmap(file_name, dtype=np.float32,
                             mode='r', shape=(rows, columns))
        self.stat = None
        if stat_file_name is not None:
            stat_file = np.load(stat_file_name)
            self.stat = np.copy(stat_file['all_var'])

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = np.copy(self.arr[idx])
        y1 = np.copy(x[1:2])
        y2 = np.copy(x[3:4])
        y = np.concatenate((y1, y2), axis=0)
        y1 = np.copy(x[0:1])
        y2 = np.copy(x[2:3])
        if self.stat is not None:
            y1 *= np.sqrt(self.stat[0])
            y2 *= np.sqrt(self.stat[2])
        y_cla = np.concatenate((y1, y2), axis=0)
        y_cla = np.rint(y_cla)
        x[0:4] = 0
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        y_cla = torch.from_numpy(y_cla.astype(int))
        return x, y, y_cla


class QQDataset(Dataset):

    def __init__(self, file_name, rows, columns, start, end, stride=1, batch_size=1, num_classes=10, stat_file_name=None):
        self.arr = np.memmap(file_name, dtype=np.float32,
                             mode='r', shape=(rows, columns))
        if (end - start) % (batch_size * stride) != 0:
            raise AttributeError("Size is not aligned.")
        self.start = start
        self.stride = stride
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.size = (end - start) // stride
        self.stat = None
        if stat_file_name is not None:
            stat_file = np.load(stat_file_name)
            self.stat = np.copy(stat_file['all_var'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find the batch index.
        batch_idx = idx // self.batch_size
        batch_offset = idx % self.batch_size
        batch_idx *= self.stride
        idx = self.start + batch_idx * self.batch_size + batch_offset

        x = np.copy(self.arr[idx])
        y = np.copy(x[0:3])
        y_cla = np.copy(y)
        if self.stat is not None:
            y_cla *= np.sqrt(self.stat[0:3])
        y_cla = np.rint(y_cla)
        y_cla[1] -= 6
        if y_cla[2] > 0:
            y_cla[2] -= 9
        y_cla[y_cla > self.num_classes - 1] = self.num_classes - 1
        x[0:3] = 0
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        y_cla = torch.from_numpy(y_cla.astype(int))
        return x, y, y_cla
