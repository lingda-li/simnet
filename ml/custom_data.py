import torch
import numpy as np
from torch.utils.data import Dataset
from cfg import data_item_format, min_complete_lat, min_store_lat, context_length, inst_length
#from cfg import input_start, target_length


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
        #y = np.concatenate((x[0:1], x[3:input_start]))
        y = np.concatenate((x[0:2], x[9:10]))
        #y = np.concatenate((x[0:2], x[3:10]))
        #y = np.concatenate((x[0:2], x[3:8], x[9:10]))
        x[0:input_start] = 0
        x = x.reshape(context_length, inst_length)
        #x = np.concatenate((x[:, 0:1], x[:, 3:inst_length]), axis=1)
        x = np.concatenate((x[:, 0:2], x[:, 9:10], x[:, input_start:inst_length]), axis=1)
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        if self.num_classes > 0:
          # FIXME
          y_cla = np.copy(y)
          #y_cla = np.concatenate((y[0:2], y[7:8]))
          y_cla = np.rint(y_cla)
          y_cla[1] -= min_complete_lat
          #if y_cla[2] > 0:
          #    y_cla[2] -= (min_store_lat - 1)
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

#from cfg_lstm import data_item_format, seq_length, inst_length, input_start, datasets

class SeqDataset(Dataset):

    def __init__(self, file_name, seqs, start, end):
        self.arr = np.memmap(file_name, dtype=data_item_format, mode='r',
                             shape=(seqs, seq_length, inst_length))
        if end <= start or end > seqs * seq_length:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start
        self.arr = self.arr.reshape((seqs * seq_length, inst_length))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += self.start
        #print('TIndex:', idx)
        if idx < seq_length:
            x = np.zeros((seq_length, inst_length - input_start))
            x[seq_length-(idx+1):seq_length, :] = np.copy(self.arr[0:idx+1, input_start:inst_length])
            #y = np.zeros((seq_length, input_start))
            #y[seq_length-(idx+1):seq_length, :] = np.copy(self.arr[0:idx+1, 0:input_start])
            y = np.copy(self.arr[idx, 0:input_start])
        else:
            x = np.copy(self.arr[idx+1-seq_length:idx+1, input_start:inst_length])
            #y = np.copy(self.arr[idx+1-seq_length:idx+1, 0:input_start])
            y = np.copy(self.arr[idx, 0:input_start])
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y


class ComSeqDataset(Dataset):

    def __init__(self, file_num, start, end):
        if file_num > len(datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += datasets[i][1]
        total_size *= seq_length
        if end <= start or end > total_size:
            raise AttributeError("End is illegal.")
        # Calculate start and end for each dataset.
        self.file_num = file_num
        self.size = end - start
        frac = self.size / total_size
        self.mm_sets = []
        self.starts = []
        self.mm_sizes = []
        self.bounds = [0]
        cum_start = 0
        cum_size = 0
        for i in range(file_num-1):
            self.starts.append(int(datasets[i][1] * seq_length * (start / total_size)))
            self.mm_sizes.append(int(datasets[i][1] * seq_length * frac))
            #print('Open', datasets[i][0], '(%d %d)' % (self.starts[i], self.mm_sizes[i]))
            self.mm_sets.append(SeqDataset(datasets[i][0], datasets[i][1],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        #print('Open', datasets[file_num-1][0], '(%d %d)' % (self.starts[file_num-1], self.mm_sizes[file_num-1]))
        self.mm_sets.append(SeqDataset(datasets[file_num-1][0], datasets[file_num-1][1],
                            self.starts[file_num-1], self.starts[file_num-1] + self.mm_sizes[file_num-1]))
        self.bounds.append(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find which set the idx falls in.
        for i in range(self.file_num):
            if idx < self.bounds[i+1]:
                #print('Index:', i, idx - self.bounds[i])
                return self.mm_sets[i].__getitem__(idx - self.bounds[i])
        raise RuntimeError("Idx is too large.")
