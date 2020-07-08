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
