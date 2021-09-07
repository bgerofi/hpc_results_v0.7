# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import h5py as h5
import numpy as np
import math
from time import sleep

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#dataset class
class CamDataset(Dataset):
  
    def init_reader(self):
        #shuffle
        if self.shuffle:
            self.rng.shuffle(self.all_files)
            
        #shard the dataset
        self.global_size = len(self.all_files)
        if self.allow_uneven_distribution and self.comm_size > 1:
            # this setting covers the data set completely, but the
            # last worker might get more samples than the rest

            idx_list = [(self.global_size + i) // self.comm_size for i in range(self.comm_size)]
            end_idx_list = [sum(idx_list[ : i + 1]) for i in range(self.comm_size)]
            start_idx_list = [end_idx_list[i] - idx_list[i] for i in range(self.comm_size)]

            assert(end_idx_list[-1] == self.global_size)

            start_idx = start_idx_list[self.comm_rank]

            if self.comm_rank != (self.comm_size - 1):
                end_idx = end_idx_list[self.comm_rank]
            else:
                end_idx = self.global_size

            self.files = self.all_files[start_idx:end_idx]

            assert(len(self.files) > 0)
            assert(len(self.files) <= self.num_local_samples)

        else:
            # here, every worker gets the same number of samples, 
            # potentially under-sampling the data
            num_files_local = self.global_size // self.comm_size
            start_idx = self.comm_rank * num_files_local
            end_idx = start_idx + num_files_local
            self.files = self.all_files[start_idx:end_idx]
            self.global_size = self.comm_size * len(self.files)

            assert(len(self.files) > 0)
            assert(len(self.files) <= self.num_local_samples)
            
        if self.padding:
            for i in range(self.num_local_samples - len(self.files)):
                assert(len(self.files) < self.num_local_samples)
                self.files.append(self.files[i])
            assert(len(self.files) == self.num_local_samples)
            
        #my own files
        self.local_size = len(self.files)

  
    def __init__(self, source, statsfile, channels, allow_uneven_distribution = False, shuffle = False, preprocess = True, padding = False, dummy = False, num_local_samples = 0, comm_size = 1, comm_rank = 0, seed = 12345):
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.padding = padding
        self.dummy = dummy
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') ] )
        self.num_local_samples = num_local_samples
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.allow_uneven_distribution = allow_uneven_distribution

        #split list of files
        self.rng = np.random.RandomState(seed)
        
        #init reader
        self.init_reader()

        #get shapes
        filename = os.path.join(self.source, self.files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape

            if self.dummy:
                self.data = fin['climate']['data'][..., self.channels]
                self.label = fin['climate']['labels_0'][...]
                self.data = np.transpose(self.data, (2,0,1))
 
        #get statsfile for normalization
        #open statsfile
        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (data_shift.shape[0], 1, 1) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (data_scale.shape[0], 1, 1) ).astype(np.float32)

        if self.dummy:
            self.data = self.data_scale * (self.data - self.data_shift)


    def __len__(self):
        return self.local_size


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.files[idx])

        if self.dummy:
            data = self.data
            label = self.label
        else:
            #load data and project
            with h5.File(filename, "r") as f:
                data = f["climate/data"][..., self.channels]
                label = f["climate/labels_0"][...]
        
            #transpose to NCHW
            data = np.transpose(data, (2,0,1))
        
            #preprocess
            data = self.data_scale * (data - self.data_shift)

        
        return data, label, filename




class CamDistributedDataset(Dataset):

    # comm_size and comm_rank are local to the given rank
    # global_size and global_rank are global across the entire MPI job
    def __init__(self, source, statsfile, channels, allow_uneven_distribution = False, shuffle = False, preprocess = True, padding = False, dummy = False, num_local_samples = 0, comm_size = 1, comm_rank = 0, seed = 12345, global_size = 1, global_rank = 0):
        self.source = source
        self.statsfile = statsfile
        self.channels = channels
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.padding = padding
        self.dummy = dummy
        self.all_files = sorted( [ os.path.join(self.source,x) for x in os.listdir(self.source) if x.endswith('.h5') ] )
        self.num_local_samples = num_local_samples
        self.comm_size = comm_size
        self.comm_rank = comm_rank
        self.global_size = global_size
        self.global_rank = global_rank
        self.allow_uneven_distribution = allow_uneven_distribution
        assert(num_local_samples == len(self.all_files))
        print("CamDistributedDataset[{}/{}]: source: {}, num_local_samples: {}".format(
            global_rank, global_size, self.source, num_local_samples))

        #split list of files
        self.rng = np.random.RandomState(seed)

        #get shapes
        filename = os.path.join(self.source, self.all_files[0])
        with h5.File(filename, "r") as fin:
            self.data_shape = fin['climate']['data'].shape
            self.label_shape = fin['climate']['labels_0'].shape

            if self.dummy:
                self.data = fin['climate']['data'][..., self.channels]
                self.label = fin['climate']['labels_0'][...]
                self.data = np.transpose(self.data, (2,0,1))

        #get statsfile for normalization
        #open statsfile
        with h5.File(self.statsfile, "r") as f:
            data_shift = f["climate"]["minval"][self.channels]
            data_scale = 1. / ( f["climate"]["maxval"][self.channels] - data_shift )

        #reshape into broadcastable shape
        self.data_shift = np.reshape( data_shift, (data_shift.shape[0], 1, 1) ).astype(np.float32)
        self.data_scale = np.reshape( data_scale, (data_scale.shape[0], 1, 1) ).astype(np.float32)

        if self.dummy:
            self.data = self.data_scale * (self.data - self.data_shift)

        # Global samples array
        self.samples = [None] * int(num_local_samples * self.global_size)
        for i in range(num_local_samples):
            self.samples[num_local_samples * self.global_rank + i] = self.all_files[i]

    def __len__(self):
        return len(self.samples)


    @property
    def shapes(self):
        return self.data_shape, self.label_shape


    def __getitem__(self, idx):
        filename = os.path.join(self.source, self.samples[idx])

        if self.dummy:
            data = self.data
            label = self.label
        else:
            #load data and project
            with h5.File(filename, "r") as f:
                data = f["climate/data"][..., self.channels]
                label = f["climate/labels_0"][...]

            #transpose to NCHW
            data = np.transpose(data, (2,0,1))

            #preprocess
            data = self.data_scale * (data - self.data_shift)


        return data, label, filename
