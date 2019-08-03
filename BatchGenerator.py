import os
from collections import defaultdict
from tqdm.autonotebook import tqdm
import numpy as np
import scipy.sparse as sp
import pickle
import torch
import torch.nn.functional as F
device = ('cuda' if torch.cuda.is_available() else 'cpu')

class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self._cursor = 0
        self.splitted = False
        self.split()
    
    def split(self, split_ratio=[0.6, 0.8]):
        ti = int(len(self.data)*split_ratio[0])
        vi = int(len(self.data)*split_ratio[1])
        self.valid_data = self.data[ti:vi]
        self.test_data = self.data[vi:]
        self.data = self.data[:ti]
        self.splitted = True
    
    def get_validation_data(self):
        for i in range(len(self.valid_data)//self.batch_size):
            batch = self.valid_data[i*self.batch_size: (i+1)*self.batch_size]
            batch_input, batch_output = self.process_batch(batch)
            yield batch_input, batch_output
    
    def get_test_data(self):
        batch_input, batch_output = self.process_batch(self.test_data)
        return batch_input, batch_output
    
    def process_batch(self, batch):
        usr = [b[0] for b in batch]
        mvi = [b[1] for b in batch]
        ratings = [b[2]-1 for b in batch]
        linkpreds = [1 if b[2] > 2 else 0 for b in batch]
        return [torch.LongTensor(usr), torch.LongTensor(mvi)], [torch.LongTensor(ratings).to(device), torch.LongTensor(linkpreds).to(device)]
    
    def __iter__(self):
        self._cursor = 0
        return self
    
    def __next__(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.data):
            batch = self.data[self._cursor:] + self.data[:ncursor - len(self.data)]
            self._cursor = ncursor - len(self.data)
            batch_input, batch_output = self.process_batch(batch)
            return batch_input, batch_output, True
        batch = self.data[self._cursor:self._cursor + self.batch_size]
        self._cursor = ncursor % len(self.data)
        batch_input, batch_output = self.process_batch(batch)
        return batch_input, batch_output, False
