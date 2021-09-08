import torch
from .spatial_local_sampler import SpatialLocalSampler
#from .distributed_image_folder import ImageFolder
from torch.utils.data import Sampler, Dataset
import numpy as np

import os.path
from mpi4py import MPI


class SpatialScheduler():
    r""" Scheduler that help to communicate the local data between subset of the dataset
    
    Args:
        dataset: Dataset used for scheduling
        sampler: Datasampler (use with spatial_local_sampler.SpatialLocalSampler)
        comm: mpi4py communicator
        non_blocking (bool, optional): If ``True`` (default), communicated based on the non-blocking mpi4py
        num_samples_comm (int, optional): Default: 3. The number of samples communicate at each iterations.
    """
    def __init__(self, dataset: None, sampler: SpatialLocalSampler, 
                    non_blocking: bool = True, num_samples_comm: int=3):
        
        self.dataset = dataset
        self.sampler = sampler
        self.non_blocking = non_blocking
        self.num_samples_comm = num_samples_comm
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.data = {}
        self.schedule = []
        self.clean_list = []
        self.is_finished = True
        if self.rank == 0:
            print(self.rank, self.size, "SpatialScheduler", num_samples_comm)
        
    def finish_schedule(self):
        return self.is_finished
    
    def clean_local_storage(self):
        for sample_idx in self.clean_list:
            self.dataset.delete_an_item(sample_idx)
        self.clean_list = []    
    
    def communicate(self, index):
        # print(self.rank,"communicate", self.num_samples_comm)
        # print(self.rank, self.schedule)
        # print(self.rank, self.data)
        
        send_requests = []
        recv_requests = []
        for i in range(0,self.num_samples_comm):
            if (self.is_finished):
                break
            sample_idx = self.schedule[len(self.schedule)-1]
            if sample_idx in self.data:
                sample_meta_data = self.data[sample_idx]
                if sample_meta_data[0] != sample_meta_data[1]:
                    if sample_meta_data[0] == self.rank:
                        self.clean_list.append(sample_idx)
                        sample, path, class_name = self.dataset.get_raw_item(sample_idx)
                        send_data = {'idx':sample_idx,'path':path,'sample':sample, 'class_name':class_name}
                        req = self.comm.isend(send_data, dest=sample_meta_data[1], tag=sample_idx)
                        #print(self.rank, "Send to", sample_meta_data[1], "sample", sample_idx)
                        send_requests.append(req)
                        
                    if sample_meta_data[1] == self.rank:
                        buf = bytearray(1<<28) # 256MB buffer (just in case)
                        req = self.comm.irecv(buf, source=sample_meta_data[0], tag=sample_idx)
                        recv_requests.append(req)
                        #print(self.rank, "wait data from", sample_meta_data[0], "sample", sample_idx)
                
            del self.schedule[len(self.schedule)-1]
            if (len(self.schedule) == 0):
                self.is_finished = True
        return send_requests, recv_requests
        
    def synchronize(self, send_requests, recv_requests):
        # print(self.rank,"synchronize")
        #r"""Check the requests finished or not"""
        # print(self.rank, "process ", send_requests, "AND", recv_requests)
        if recv_requests is not None and len(recv_requests) > 0:
            for req in recv_requests:
                data = req.wait()
                #print(req, data)
                self.dataset.add_a_item(data['idx'],data['path'],data['class_name'],data['sample'])
                #print(self.rank, "finish add item", data['idx'])
                
        if send_requests is not None and len(send_requests) > 0:
            for req in send_requests:
                req.wait()

        # print(self.rank,"Start Barrier")
        self.comm.Barrier()
    
    def scheduling(self):
        #print(self.rank,"scheduling")
        this_indices_global = self.sampler.get_this_global_indices_list()  # A list global index in each rank...
        next_indices_global = self.sampler.get_next_global_indices_list()
        assert self.size == len(this_indices_global)
        assert self.size == len(next_indices_global)
        
        #print(self.rank, "this_indices_global", this_indices_global)
        #print(self.rank, "next_indices_global", next_indices_global)
        
        self.data = {}
        for source_idx in range(0,len(this_indices_global)):
            for sample_index in this_indices_global[source_idx]:
                self.data[sample_index] = [source_idx, None]

        for dest_idx in range(0,len(next_indices_global)):
            for sample_index in next_indices_global[dest_idx]:
                self.data[sample_index][1] = dest_idx

        self.schedule = []
        indices_global_all = []
        for i in range(0, len(next_indices_global)):
            indices_global_all = indices_global_all + next_indices_global[i]

        for i in range(0, len(next_indices_global[0])): #Assume all item has the same length. Need check
            self.schedule = self.schedule + indices_global_all[i:len(indices_global_all):len(next_indices_global[0])]
                
        self.is_finished = False
        self.clean_list = []
        # print(self.rank, "schedule", self.schedule)
        # print(self.rank, "data", self.data)
