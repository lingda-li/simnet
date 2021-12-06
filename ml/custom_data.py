import torch
import numpy as np
from torch.utils.data import Dataset
from cfg import data_item_format, min_complete_lat, min_store_lat, context_length, inst_length, input_start, target_length


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
        self.arr = np.memmap(file_name, dtype=data_item_format,
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
        y_cla[1] -= min_complete_lat
        if y_cla[2] > 0:
            y_cla[2] -= (min_store_lat - 1)
        y_cla[y_cla > self.num_classes - 1] = self.num_classes - 1
        x[0:3] = 0
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        y_cla = torch.from_numpy(y_cla.astype(int))
        return x, y, y_cla


class CompressedDataset(Dataset):

    def __init__(self, file_name, rows, insts, start, end, stride=1, batch_size=1, num_classes=10, stat_file_name=None):
        self.idx = np.memmap(file_name + '.idx', dtype=np.uint64,
                             mode='r', shape=rows)
        self.arr = np.memmap(file_name + '.dat', dtype=data_item_format,
                             mode='r', shape=insts*inst_length)
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
        if self.stride != 1:
            batch_idx = idx // self.batch_size
            batch_offset = idx % self.batch_size
            batch_idx *= self.stride
            idx = batch_idx * self.batch_size + batch_offset
        idx += self.start

        start_idx = self.idx[idx]
        end_idx = self.idx[idx+1]
        assert (end_idx - start_idx) % inst_length == 0 and end_idx - start_idx <= context_length*inst_length
        x = np.zeros(context_length*inst_length)
        x[0:end_idx-start_idx] = np.copy(self.arr[start_idx:end_idx])
        #y = np.copy(x[0:target_length])
        y = np.concatenate((x[0:1], x[3:input_start]))
        x[0:input_start] = 0
        x = x.reshape(context_length, inst_length)
        x = np.concatenate((x[:, 0:1], x[:, 3:inst_length]), axis=1)
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        if self.num_classes > 0:
          y_cla = np.copy(y)
          y_cla = np.rint(y_cla)
          # FIXME
          y_cla[1] -= min_complete_lat
          if y_cla[2] > 0:
              y_cla[2] -= (min_store_lat - 1)
          y_cla[y_cla > self.num_classes - 1] = self.num_classes - 1
          y_cla = torch.from_numpy(y_cla.astype(int))
          return x, y, y_cla
        else:
          return x, y


class CombinedDataset(Dataset):

    def __init__(self, start, end, num_classes=10):
        self.size = end - start
        if self.size % 3 != 0 or start % 3 != 0:
            raise AttributeError("Size is not aligned.")
        size = self.size // 3
        start = start // 3
        self.dat0 = CompressedDataset("data_spec_robreg/p128r120/all.qqu", 81939571, 4304420921, start, start + size, num_classes=num_classes)
        self.dat1 = CompressedDataset("data_spec_robreg/p128r80/all.qqu", 80764702, 4190464494, start, start + size, num_classes=num_classes)
        self.dat2 = CompressedDataset("data_spec_robreg/p128r40/all.qqu", 71433975, 3456072209, start, start + size, num_classes=num_classes)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sub_idx = idx // 3
        if idx % 3 == 0:
            item = self.dat0.__getitem__(sub_idx)
            return item[0], item[1], item[2], torch.tensor([2.])
        elif idx % 3 == 1:
            item = self.dat1.__getitem__(sub_idx)
            return item[0], item[1], item[2], torch.tensor([1.])
        else:
            item = self.dat2.__getitem__(sub_idx)
            return item[0], item[1], item[2], torch.tensor([0.])
